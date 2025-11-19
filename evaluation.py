# evaluation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


def focal_loss_with_logits(logits, labels, alpha, gamma):
    """
    logits: (B, K)
    labels: (B, K) 0/1

    返回逐元素 focal loss (B, K)，尚未做 mask 和平均

    公式：
      bce = BCEWithLogitsLoss(reduction="none")
      p   = sigmoid(logits)
      pt  = p*y + (1-p)*(1-y)
      focal = alpha * (1-pt)^gamma * bce
    """
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        reduction="none",
    )  # (B, K)

    p = torch.sigmoid(logits)  # (B, K)
    pt = p * labels + (1 - p) * (1 - labels)
    focal_weight = alpha * (1.0 - pt).pow(gamma)

    loss = focal_weight * bce
    return loss  # (B, K)


def compute_loss(logits, labels, sent_mask, loss_type="bce", pos_weight=None):
    """
    logits: (B, K)
    labels: (B, K)
    sent_mask: (B, K)

    返回标量 loss
    """
    if loss_type == "bce":
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss(reduction="none")
        loss_raw = criterion(logits, labels.float())   # (B, K)

    elif loss_type == "focal":
        alpha = getattr(config, "FOCAL_ALPHA", 0.25)
        gamma = getattr(config, "FOCAL_GAMMA", 2.0)
        loss_raw = focal_loss_with_logits(
            logits,
            labels,
            alpha=alpha,
            gamma=gamma,
        )  # (B, K)

    else:
        raise NotImplementedError("unknown loss_type: %s" % loss_type)

    loss = (loss_raw * sent_mask).sum() / sent_mask.sum()
    return loss


def compute_stats_at_tau(probs_flat, labels_flat, tau):
    """
    在给定阈值 tau 下算 acc / precision / recall / f1。
    probs_flat: (N,)
    labels_flat: (N,) 0/1
    """
    preds_flat = (probs_flat >= tau).long()

    N = labels_flat.size(0)
    correct = (preds_flat == labels_flat).sum().item()
    acc = correct / float(N)

    tp = ((preds_flat == 1) & (labels_flat == 1)).sum().item()
    fp = ((preds_flat == 1) & (labels_flat == 0)).sum().item()
    fn = ((preds_flat == 0) & (labels_flat == 1)).sum().item()

    eps = 1e-8
    precision = tp / float(tp + fp + eps)
    recall    = tp / float(tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    stats = {
        "tau": tau,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "N": N,
        "pos": int((labels_flat == 1).sum().item()),
        "neg": int((labels_flat == 0).sum().item()),
        "pred_pos": int((preds_flat == 1).sum().item()),
        "pred_neg": int((preds_flat == 0).sum().item()),
    }
    return stats


def sweep_best_tau(probs_flat, labels_flat, tau_list=None):
    """
    在一组候选 tau 里扫 F1，找 F1 最大的那个。
    返回:
      best_tau, best_stats
    """
    if tau_list is None:
        # 你可以自己改这里的范围和步长
        tau_list = [i / 100.0 for i in range(5, 96, 5)]  # 0.05, 0.10, ..., 0.95

    best_tau = None
    best_stats = None
    best_f1 = -1.0

    for tau in tau_list:
        stats = compute_stats_at_tau(probs_flat, labels_flat, tau)
        if stats["f1"] > best_f1:
            best_f1 = stats["f1"]
            best_f1 = stats["f1"]
            best_tau = tau
            best_stats = stats

    return best_tau, best_stats


def evaluate(encoder, feat_builder, classifier, dataloader, device):
    """
    在一个 dataloader 上跑完整个 dev/test：
      - 算平均 loss
      - 在 tau=0.5 下的 acc / precision / recall / f1
      - 在一组 tau 值里扫 best f1 和对应指标
    """
    encoder.eval()
    feat_builder.eval()
    classifier.eval()

    total_loss = 0.0
    n_batches = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sent_spans     = batch["sent_spans"]
            sent_mask      = batch["sent_mask"].to(device)
            labels         = batch["labels"].to(device)

            meta = {
                "roles":    batch["roles"].to(device),
                "sent_len": batch["sent_len"].to(device),
                "position": batch["position"].to(device),
            }

            # 句向量
            H, sent_mask_enc = encoder(input_ids, attention_mask, sent_spans, sent_mask)

            # 先跑一下 feature（边界版暂时不用）
            _ = feat_builder(meta, sent_mask_enc)

            # 如果这一 batch 的对话都只有 1 句，就没有边界，跳过
            if H.size(1) <= 1:
                continue

            # 边界向量: (B, K-1, 4*d_enc_sent)
            H_prev = H[:, :-1, :]
            H_cur  = H[:, 1:, :]

            H_diff    = H_cur - H_prev
            H_absdiff = H_diff.abs()

            H_edge = torch.cat(
                [H_prev, H_cur, H_diff, H_absdiff],
                dim=-1,
            )

            # 边界标签和 mask
            labels_edge = labels[:, 1:]                      # (B, K-1)
            sent_mask_edge = sent_mask_enc[:, 1:] * sent_mask_enc[:, :-1]

            if sent_mask_edge.sum() == 0:
                continue

            logits = classifier(H_edge, sent_mask_edge, F=None)  # (B, K-1)

            loss = compute_loss(
                logits,
                labels_edge,
                sent_mask_edge,
                loss_type=config.LOSS_TYPE,
                pos_weight=None,
            )
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)          # (B, K-1)
            mask  = sent_mask_edge > 0.5          # (B, K-1)
            all_probs.append(probs[mask].cpu())
            all_labels.append(labels_edge[mask].cpu())

    # 平均 loss
    if n_batches == 0:
        avg_loss = 0.0
    else:
        avg_loss = total_loss / n_batches

    if len(all_probs) == 0:
        return {
            "loss": avg_loss,
            "tau": 0.5,
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tau_best": 0.5,
            "f1_best": 0.0,
        }

    probs_flat  = torch.cat(all_probs, dim=0)   # (N_edge,)
    labels_flat = torch.cat(all_labels, dim=0)  # (N_edge,)

    # τ=0.5 的指标
    stats_05 = compute_stats_at_tau(probs_flat, labels_flat, tau=0.5)

    # 在一组 tau 里面扫 best f1
    tau_list = [0.05 * i for i in range(1, 20)]
    best_tau, best_stats = sweep_best_tau(probs_flat, labels_flat, tau_list)

    # 打印几个 summary
    N = labels_flat.numel()
    pos = int(labels_flat.sum().item())
    neg = N - pos
    pred_pos_05 = int((probs_flat >= 0.5).sum().item())
    pred_neg_05 = N - pred_pos_05

    print(
        f"[EVAL] tau=0.5: N={N} pos={pos} neg={neg} "
        f"pred_pos={pred_pos_05} pred_neg={pred_neg_05} "
        f"tp={stats_05['tp']} fp={stats_05['fp']} fn={stats_05['fn']}"
    )

    N_b  = N
    pos_b = pos
    neg_b = neg
    tp_b  = best_stats["tp"]
    fp_b  = best_stats["fp"]
    fn_b  = best_stats["fn"]
    tn_b  = N_b - tp_b - fp_b - fn_b

    print(
        "[EVAL] Confusion matrix at best_tau "
        f"({best_tau:.2f}):\n"
        f"         pred=0    pred=1\n"
        f"true=0   {tn_b:6d}   {fp_b:6d}\n"
        f"true=1   {fn_b:6d}   {tp_b:6d}"
    )

    metrics = {
        "loss": avg_loss,
        "tau": 0.5,
        "acc": stats_05["acc"],
        "precision": stats_05["precision"],
        "recall": stats_05["recall"],
        "f1": stats_05["f1"],
        "tau_best": best_tau,
        "f1_best": best_stats["f1"],
        "precision_best": best_stats["precision"],
        "recall_best": best_stats["recall"],
    }
    return metrics

