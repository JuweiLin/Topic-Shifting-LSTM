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

        # 切片：忽略第0句
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(config.ENCODER_NAME, use_fast=True)

    # 保证 SENT_TOKEN 在 vocab 里（用于句子边界）
    if hasattr(config, "SENT_TOKEN") and config.SENT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [config.SENT_TOKEN]})
        print(f"Added SENT_TOKEN '{config.SENT_TOKEN}' to tokenizer.")


    train_loader = build_dataloader(
        [config.TRAIN_PATH, config.TEST_PATH],   # 这里传 list
        tokenizer,
        batch_size_dialog=config.BATCH_SIZE_DIALOG,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    # 2) 验证：用 test 当 dev
    dev_loader = build_dataloader(
        config.DEV_PATH,                      # 用 test.json 做验证
        tokenizer,
        batch_size_dialog=config.BATCH_SIZE_DIALOG,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    encoder = HFEncoder(
        model_name=config.ENCODER_NAME,
        unfreeze_top=config.UNFREEZE_TOP,
    ).to(device)
    # 如果 tokenizer vocab 大小变了，扩展 encoder 的词表
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

    # 处理 POS_WEIGHT 为 None 的情况
    if config.POS_WEIGHT is not None:
        pos_weight = torch.tensor(config.POS_WEIGHT, device=device)
        print(f"Using POS_WEIGHT: {config.POS_WEIGHT}")
    else:
        pos_weight = None
        print("Using NO POS_WEIGHT (Standard BCE)")

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


if __name__ == "__main__":
    main()