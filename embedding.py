import torch
import torch.nn as nn
from transformers import AutoModel


class HFEncoder(nn.Module):
    def __init__(self, model_name="FacebookAI/roberta-base", unfreeze_top=8):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self._freeze_layers(unfreeze_top)
        # 注意力池化头：给每个 token 打一个分数
        self.sent_pool = nn.Linear(self.hidden_size, 1)

    @staticmethod
    def _get_transformer_layers(model):
        """
        通用地拿到 encoder 的 block 列表，兼容 Roberta / DeBERTa / BART 等
        """
        # Roberta / BERT: model.encoder.layer
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer

        # DeBERTa-v2/v3: model.deberta.encoder.layer
        if (
            hasattr(model, "deberta")
            and hasattr(model.deberta, "encoder")
            and hasattr(model.deberta.encoder, "layer")
        ):
            return model.deberta.encoder.layer

        # BART: model.model.encoder.layers
        if (
            hasattr(model, "model")
            and hasattr(model.model, "encoder")
            and hasattr(model.model.encoder, "layers")
        ):
            return model.model.encoder.layers

        # 找不到就返回 None
        return None

        
    def _freeze_layers(self, unfreeze_top):
        """
        unfreeze_top 语义：
          - -1: 全部解冻（不冻结）
          - 0: 全部冻结（只训练后面的分类头）
          - >0: 解冻最后 N 层 encoder block
        """
        layers = self._get_transformer_layers(self.model)

        # 先全部冻结
        for p in self.model.parameters():
            p.requires_grad = False

        # -1 表示全部解冻
        if unfreeze_top == -1:
            for p in self.model.parameters():
                p.requires_grad = True
            return

        # 没找到层，或者 unfreeze_top == 0，就保持全冻结
        if layers is None or unfreeze_top <= 0:
            return

        # 解冻最后 N 层
        total = len(layers)
        keep_from = max(0, total - unfreeze_top)
        for i, block in enumerate(layers):
            if i >= keep_from:
                for p in block.parameters():
                    p.requires_grad = True

        # 如果有一些最终的 LayerNorm / pooler，可以顺带解冻（可选）
        for attr in ["layernorm", "LayerNorm", "ln_f", "final_layer_norm", "pooler"]:
            mod = getattr(self.model, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True

    def forward(self, input_ids, attention_mask, sent_spans, sent_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        token_hidden = outputs.last_hidden_state  # (B, L, hidden_size)

        H, sent_mask_out = mean_pool_by_spans(
            token_hidden,
            sent_spans,
            attention_mask=attention_mask,
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
