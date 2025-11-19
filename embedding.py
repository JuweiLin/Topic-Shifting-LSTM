import torch
import torch.nn as nn
from transformers import AutoModel


class QwenSentenceEncoder(nn.Module):
    def __init__(self, model_name="FacebookAI/roberta-base", unfreeze_top=8):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self._freeze_layers(unfreeze_top)
        # 注意力池化头：给每个 token 打一个分数
        self.sent_pool = nn.Linear(self.hidden_size, 1)
        
    def _freeze_layers(self, unfreeze_top):
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            return

        blocks = self.model.model.layers
        n = len(blocks)
        keep_from = max(0, n - unfreeze_top)

        for p in self.model.parameters():
            p.requires_grad = False

        for i, block in enumerate(blocks):
            if i >= keep_from:
                for p in block.parameters():
                    p.requires_grad = True

    def forward(self, input_ids, attention_mask, sent_spans, sent_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_hidden = outputs.last_hidden_state  # (B, L, hidden_size)

        # 这里直接用注意力池化，不保留原来的 mean pooling
        H, sent_mask_out = mean_pool_by_spans(
            token_hidden,
            sent_spans,
            attention_mask,
            sent_pool=self.sent_pool,
        )
        return H, sent_mask_out


def mean_pool_by_spans(
    token_hidden,
    sent_spans,
    attention_mask=None,
    max_K=None,
    sent_pool=None,
):
    """
    现在这个函数只做注意力池化（如果 sent_pool 为 None 会报错，正常情况总是传 self.sent_pool 进来）
    """
    B, L, hidden_size = token_hidden.size()

    if max_K is None:
        max_K = 0
        for spans in sent_spans:
            if len(spans) > max_K:
                max_K = len(spans)
    if max_K == 0:
        H = torch.zeros(
            B, 1, hidden_size,
            device=token_hidden.device,
            dtype=token_hidden.dtype,
        )
        sent_mask = torch.zeros(B, 1, device=token_hidden.device)
        return H, sent_mask

    H = torch.zeros(
        B, max_K, hidden_size,
        device=token_hidden.device,
        dtype=token_hidden.dtype,
    )
    sent_mask = torch.zeros(B, max_K, device=token_hidden.device)

    if sent_pool is None:
        raise ValueError("sent_pool must be provided for attention pooling")

    for b in range(B):
        spans = sent_spans[b]
        for k, span in enumerate(spans):
            if k >= max_K:
                break
            start, end = span
            start = int(start)
            end = int(end)
            if end <= start or start >= L:
                continue
            end = min(end, L)

            vecs = token_hidden[b, start:end, :]  # (len, H)

            # 1. 每个 token 一个 score: (len,)
            scores = sent_pool(vecs).squeeze(-1)  # (len,)

            # 2. 如果有 attention_mask，就把 padding 的位置屏蔽掉
            if attention_mask is not None:
                am = attention_mask[b, start:end]  # (len,)
                am = am.float()
                if am.sum().item() == 0:
                    # 这一句实际上没有有效 token，跳过，保持 mask=0
                    continue
                scores = scores.masked_fill(am < 0.5, float("-inf"))

            # 3. softmax 得到权重
            weights = torch.softmax(scores, dim=0)    # (len,)

            # 4. 加权求和得到句向量
            pooled = (vecs * weights.unsqueeze(-1)).sum(dim=0)  # (H,)

            H[b, k, :] = pooled
            sent_mask[b, k] = 1.0

    return H, sent_mask
