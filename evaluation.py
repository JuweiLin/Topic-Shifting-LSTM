# evaluation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p   = torch.sigmoid(logits)
    pt  = torch.where(targets > 0.5, p, 1.0 - p)
    loss = alpha * (1.0 - pt) ** gamma * bce
    return loss


def compute_loss(logits, labels, sent_mask, loss_type="bce", pos_weight=None):
    mask = sent_mask > 0.5
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    logits_flat = logits[mask]
    labels_flat = labels[mask].float()

    if loss_type == "bce":
        if pos_weight is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits_flat, labels_flat)
        return loss

    elif loss_type == "focal":
        alpha = getattr(config, "FOCAL_ALPHA", 0.75)
        gamma = getattr(config, "FOCAL_GAMMA", 2.0)
        loss_vec = focal_loss_with_logits(logits_flat, labels_flat, alpha=alpha, gamma=gamma)
        return loss_vec.mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_stats_at_tau(probs_flat, labels_flat, tau):
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
    }
    return stats


def sweep_best_tau(probs_flat, labels_flat, tau_list=None, step=0.01):
    """
    扫描不同 tau 找 F1 最优的阈值
    - 默认: 从 0.00 到 1.00, 步长 step
    - 也可以直接传 tau_list 覆盖
    """
    if tau_list is None:
        n_steps = int(1.0 / step)
        tau_list = [i * step for i in range(0, n_steps + 1)]

    best_tau = 0.5
    best_stats = None
    best_f1 = -1.0

    for tau in tau_list:
        stats = compute_stats_at_tau(probs_flat, labels_flat, tau)
        f1 = stats["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_stats = stats

    if best_stats is None:
        best_stats = compute_stats_at_tau(probs_flat, labels_flat, 0.5)

    return best_tau, best_stats




def evaluate(encoder, feat_builder, classifier, dataloader, device):
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
            
            logits = logits_all[:, 1:]            
            labels_edge = labels[:, 1:]           
            sent_mask_edge = sent_mask_enc[:, 1:] 

            if sent_mask_edge.sum() == 0:
                continue

            loss = compute_loss(
                logits,
                labels_edge,
                sent_mask_edge,
                loss_type=config.LOSS_TYPE,
                pos_weight=None,
            )
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)           
            mask  = sent_mask_edge > 0.5           
            all_probs.append(probs[mask].cpu())
            all_labels.append(labels_edge[mask].cpu())

    if n_batches == 0:
        avg_loss = 0.0
    else:
        avg_loss = total_loss / n_batches

    if len(all_probs) == 0:
        return {"loss": avg_loss, "f1": 0.0}

    probs_flat  = torch.cat(all_probs, dim=0)   
    labels_flat = torch.cat(all_labels, dim=0)  

    # ==========================================

    stats_05 = compute_stats_at_tau(probs_flat, labels_flat, tau=0.5)

    # 用细粒度阈值从 0~1 扫一遍，步长 0.01
    best_tau, best_stats = sweep_best_tau(
        probs_flat,
        labels_flat,
        tau_list=None,  # 不传就用 step 生成 [0.00, 0.01, ..., 1.00]
        step=0.01
    )


    N = labels_flat.numel()
    pos = int(labels_flat.sum().item())
    neg = N - pos
    
    tp_b = best_stats["tp"]
    fp_b = best_stats["fp"]
    fn_b = best_stats["fn"]
    tn_b = N - tp_b - fp_b - fn_b

    print(
        f"[EVAL] Best Tau={best_tau:.2f} | F1={best_stats['f1']:.4f}\n"
        f"       Confusion Matrix:\n"
        f"                 Pred=0    Pred=1\n"
        f"       True=0   {tn_b:6d}   {fp_b:6d}\n"
        f"       True=1   {fn_b:6d}   {tp_b:6d}"
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
    }
    return metrics