import torch
import torch.nn as nn
from transformers import AutoTokenizer

from load_tiage import build_dataloader
from embedding import QwenSentenceEncoder
from feature import SentenceFeatureBuilder
from classifier import SentenceShiftClassifier
import config
from evaluation import compute_loss, evaluate


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

        meta = {}
        if isinstance(config.FEATURE_SET, str):
            feat_names = [x.strip() for x in config.FEATURE_SET.split(",") if x.strip()]
        else:
            feat_names = list(config.FEATURE_SET)

        for name in feat_names:
            if name in batch:
                meta[name] = batch[name].to(device)

        # 句向量 H: (B, K, d_enc_sent)
        H, sent_mask_enc = encoder(input_ids, attention_mask, sent_spans, sent_mask)

        # 先算一下句子特征（现在边界版里先不用，预留位置）
        _ = feat_builder(meta, sent_mask_enc)

        # ========== 边界表示：u(i-1) -> u(i) ==========
        # 如果整段对话只有一个句子，就没有边界可以学，直接跳过这个 batch
        if H.size(1) <= 1:
            continue

        # H_prev / H_cur: (B, K-1, d_enc_sent)
        H_prev = H[:, :-1, :]
        H_cur  = H[:, 1:, :]

        H_diff    = H_cur - H_prev
        H_absdiff = H_diff.abs()

        # H_edge: (B, K-1, 4 * d_enc_sent)
        H_edge = torch.cat(
            [H_prev, H_cur, H_diff, H_absdiff],
            dim=-1,
        )

        # 边界标签：用“当前句”的标签，表示 u(i-1) -> u(i) 这条边
        labels_edge = labels[:, 1:]                # (B, K-1)

        # 边界 mask：前后两句都存在才是有效边
        sent_mask_edge = sent_mask_enc[:, 1:] * sent_mask_enc[:, :-1]   # (B, K-1)

        # 极端情况：这个 batch 里实际上没有有效边（mask 全 0），就跳过
        if sent_mask_edge.sum() == 0:
            continue

        # 边界级分类器：不再用句子特征，F 传 None
        logits = classifier(H_edge, sent_mask_edge, F=None)  # (B, K-1)

        # 用边界标签和边界 mask 算 loss
        loss = compute_loss(
            logits,
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

        if i == 0:
            print(
                "H_edge shape:", H_edge.shape,
                "logits shape:", logits.shape,
                "labels_edge shape:", labels_edge.shape,
            )

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

    tokenizer = AutoTokenizer.from_pretrained(config.ENCODER_NAME)

    train_loader = build_dataloader(
        config.TRAIN_PATH,
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

    encoder = QwenSentenceEncoder(
        model_name=config.ENCODER_NAME,
        unfreeze_top=config.UNFREEZE_TOP,
    ).to(device)

    feat_builder = SentenceFeatureBuilder(
        feature_set=config.FEATURE_SET
    ).to(device)

    # 单句向量维度
    d_enc_sent = encoder.hidden_size

    # 边界向量: [prev, cur, diff, |diff|] → 4 * d_enc_sent
    d_enc = d_enc_sent * 4

    # 目前边界版暂时不拼句子 feature，保持 0
    d_feat = 0

    classifier = SentenceShiftClassifier(
        d_enc=d_enc,
        d_feat=d_feat,
        backbone=config.BACKBONE,
    ).to(device)



    params = (
        list(encoder.parameters())
        + list(feat_builder.parameters())
        + list(classifier.parameters())
    )
    optimizer = torch.optim.AdamW(
        params,
        lr=config.LR_HEAD,
        weight_decay=config.WEIGHT_DECAY,
    )

    pos_weight = torch.tensor(config.POS_WEIGHT, device=device)


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
