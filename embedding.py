import torch
import torch.nn as nn
from transformers import AutoModel


class QwenSentenceEncoder(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2-4B-Instruct", unfreeze_top=8):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self._freeze_layers(unfreeze_top)

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

        H, sent_mask_out = mean_pool_by_spans(token_hidden, sent_spans, attention_mask)
        return H, sent_mask_out


def mean_pool_by_spans(token_hidden, sent_spans, attention_mask=None, max_K=None):
    B, L, hidden_size = token_hidden.size()

    if max_K is None:
        max_K = 0
        for spans in sent_spans:
            if len(spans) > max_K:
                max_K = len(spans)
    if max_K == 0:
        H = torch.zeros(B, 1, hidden_size, device=token_hidden.device, dtype=token_hidden.dtype)
        sent_mask = torch.zeros(B, 1, device=token_hidden.device)
        return H, sent_mask

    H = torch.zeros(B, max_K, hidden_size, device=token_hidden.device, dtype=token_hidden.dtype)
    sent_mask = torch.zeros(B, max_K, device=token_hidden.device)

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

            vecs = token_hidden[b, start:end, :]
            if attention_mask is not None:
                am = attention_mask[b, start:end].unsqueeze(-1)
                am = am.float()
                mask_sum = am.sum()
                if mask_sum.item() > 0:
                    vecs = vecs * am
                    pooled = vecs.sum(dim=0) / mask_sum
                else:
                    pooled = vecs.mean(dim=0)
            else:
                pooled = vecs.mean(dim=0)

            H[b, k, :] = pooled
            sent_mask[b, k] = 1.0

    return H, sent_mask
