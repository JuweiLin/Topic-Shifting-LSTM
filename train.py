# train.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer

import random
import numpy as np

from load_tiage import build_dataloader
from embedding import HFEncoder
from feature import SentenceFeatureBuilder
from classifier import SentenceShiftClassifier
import config
from evaluation import compute_loss, evaluate
import os
import json

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)



def train_one_epoch(
    encoder,
    feat_builder,
    classifier,
    dataloader,
    optimizer,
    device,
    loss_type,
    pos_weight,
    grad_accum_steps=1,
):
    encoder.train()
    feat_builder.train()
    classifier.train()

    total_loss = 0.0
    step = 0

    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sent_spans     = batch["sent_spans"]
        sent_mask      = batch["sent_mask"].to(device)
        labels         = batch["labels"].to(device)


        # ==========================================

        meta = {}
        if isinstance(config.FEATURE_SET, str):
            feat_names = [x.strip() for x in config.FEATURE_SET.split(",") if x.strip()]
        else:
            feat_names = list(config.FEATURE_SET)

        for name in feat_names:
            if name in batch:
                meta[name] = batch[name].to(device)

        H, sent_mask_enc = encoder(input_ids, attention_mask, sent_spans, sent_mask)
        feat_out = feat_builder(meta, sent_mask_enc)

        if H.size(1) <= 1:
            continue
            
        logits_all = classifier(H, sent_mask_enc, F=feat_out) 

        logits_edge = logits_all[:, 1:]        
        labels_edge = labels[:, 1:]            
        sent_mask_edge = sent_mask_enc[:, 1:]  

        if sent_mask_edge.sum() == 0:
            continue

        loss = compute_loss(
            logits_edge,
            labels_edge,
            sent_mask_edge,
            loss_type=loss_type,
            pos_weight=pos_weight,
        )

        loss = loss / float(grad_accum_steps)
        loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            total_loss += loss.item()

    if step == 0:
        return 0.0
    return total_loss / step


def main():
    set_seed(config.SEED)
    ckpt_path = getattr(config, "BEST_MODEL_PATH", "ckpt_tiage_roberta_best.pt")
    log_path  = getattr(config, "TRAIN_LOG_PATH", "train_log_tiage.jsonl")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(config.ENCODER_NAME, use_fast=True)

    if hasattr(config, "SENT_TOKEN") and config.SENT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [config.SENT_TOKEN]})
        print(f"Added SENT_TOKEN '{config.SENT_TOKEN}' to tokenizer.")


    train_loader = build_dataloader(
        [config.TRAIN_PATH, config.TEST_PATH],
        tokenizer,
        batch_size_dialog=config.BATCH_SIZE_DIALOG,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    dev_loader = build_dataloader(
        config.DEV_PATH,
        tokenizer,
        batch_size_dialog=config.BATCH_SIZE_DIALOG,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    encoder = HFEncoder(
        model_name=config.ENCODER_NAME,
        unfreeze_top=config.UNFREEZE_TOP,
    ).to(device)
    if hasattr(encoder, "model") and hasattr(encoder.model, "resize_token_embeddings"):
        encoder.model.resize_token_embeddings(len(tokenizer))
        print("Resized token embeddings to", len(tokenizer))


    feat_builder = SentenceFeatureBuilder(
        feature_set=config.FEATURE_SET
    ).to(device)

    d_enc_input = encoder.hidden_size
    d_feat_dim = feat_builder.get_dim()

    classifier = SentenceShiftClassifier(
        d_enc=d_enc_input,       
        d_feat=d_feat_dim,       
        backbone=config.BACKBONE,
        hidden_size=256,
        cnn_filters=getattr(config, "CNN_FILTERS", 100),
        cnn_kernel_sizes=tuple(getattr(config, "CNN_KERNEL_SIZES", [2, 3])),
    ).to(device)


    print(f"Model initialized. Backbone: {config.BACKBONE}")

    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    other_params = list(feat_builder.parameters()) + list(classifier.parameters())

    print(f"Encoder trainable params: {sum(p.numel() for p in enc_params)}")
    print(f"Head+Feature trainable params: {sum(p.numel() for p in other_params)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": config.LR_ENC},
            {"params": other_params, "lr": config.LR_HEAD},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    if config.POS_WEIGHT is not None:
        pos_weight = torch.tensor(config.POS_WEIGHT, device=device)
        print(f"Using POS_WEIGHT: {config.POS_WEIGHT}")
    else:
        pos_weight = None
        print("Using NO POS_WEIGHT (Standard BCE)")
    best_f1 = -1.0
    for epoch in range(config.EPOCHS):
        print("=== Epoch", epoch, "===")
        train_loss = train_one_epoch(
            encoder,
            feat_builder,
            classifier,
            train_loader,
            optimizer,
            device,
            config.LOSS_TYPE,
            pos_weight,
            grad_accum_steps=config.GRAD_ACCUM_STEPS,
        )
        print("train_loss:", train_loss)

        metrics = evaluate(
            encoder,
            feat_builder,
            classifier,
            dev_loader,
            device,
        )
        print("dev metrics:", metrics)
        log_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "dev_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in metrics.items()},
        }
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] Failed to write log to {log_path}: {e}")

        # save best model
        cur_f1 = metrics.get("f1_best", metrics.get("f1", 0.0))
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            print(f"[CKPT] New best f1_best={best_f1:.4f}, saving model to {ckpt_path}")

            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "encoder_state": encoder.state_dict(),
                    "feat_state": feat_builder.state_dict(),
                    "clf_state": classifier.state_dict(),
                    "config": {k: getattr(config, k) for k in dir(config) if k.isupper()},
                },
                ckpt_path,
            )

if __name__ == "__main__":
    main()