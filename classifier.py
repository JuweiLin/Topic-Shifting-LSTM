# classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_sizes, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for k in kernel_sizes:
            conv = nn.Conv1d(
                in_channels=input_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding="same",
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(num_filters))
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C_in, K)
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            o = conv(x)       # (B, F, K)
            o = bn(o)
            o = F.relu(o)
            o = self.dropout(o)
            outs.append(o)
        # 通道维拼接 -> (B, F * len(kernel_sizes), K)
        return torch.cat(outs, dim=1)


class SentenceShiftClassifier(nn.Module):
    def __init__(
        self,
        d_enc,
        d_feat=0,
        backbone="lstm",        # "none" / "rnn" / "lstm" / "gru" / "cnn"
        hidden_size=512,
        num_layers=2,
        dropout=0.2,
        cnn_filters=200,        # 每个 kernel 的通道数，相当于 CNN hidden_size
        cnn_kernel_sizes=(2,3,4),
        head_hidden=256,        # 句级 MLP head 的隐藏维度
        **kwargs,
    ):
        super().__init__()
        self.backbone_type = backbone
        d_in = d_enc + d_feat

        self.backbone = None
        self.cnn_stage1 = None
        self.cnn_stage2 = None
        d_backbone_out = d_in

        # 1) 不做序列建模
        if backbone == "none":
            self.backbone = None
            d_backbone_out = d_in

        # 2) RNN / LSTM / GRU
        elif backbone == "rnn":
            self.backbone = nn.RNN(
                input_size=d_in,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            d_backbone_out = 2 * hidden_size

        elif backbone == "lstm":
            self.backbone = nn.LSTM(
                input_size=d_in,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            d_backbone_out = 2 * hidden_size

        elif backbone == "gru":
            self.backbone = nn.GRU(
                input_size=d_in,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            d_backbone_out = 2 * hidden_size

        # 3) CNN: LeNet 风格的 1D CNN（stage1 + stage2）
        elif backbone == "cnn":
            # Stage 1: 多尺度卷积（你现在已经在用的 MultiScaleCNN）
            self.cnn_stage1 = MultiScaleCNN(
                input_dim=d_in,
                num_filters=cnn_filters,
                kernel_sizes=cnn_kernel_sizes,
                dropout=dropout,
            )
            c1 = cnn_filters * len(cnn_kernel_sizes)   # stage1 输出通道数

            # Stage 2: 再来一层 1D Conv block，扩大一点感受野
            self.cnn_stage2 = nn.Sequential(
                nn.Conv1d(c1, c1, kernel_size=3, padding=1),
                nn.BatchNorm1d(c1),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            d_backbone_out = c1   # 最终每句的 hidden 维度

        else:
            raise ValueError("Unknown backbone: %s" % backbone)

        # 归一化 + Dropout + 两层 MLP head
        self.ln = nn.LayerNorm(d_backbone_out)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_backbone_out, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, H, sent_mask, F=None):
        """
        H: (B, K, d_enc)
        F: (B, K, d_feat) or None
        sent_mask: (B, K)  1 for real sentence, 0 for pad
        """
        if F is not None:
            X = torch.cat([H, F], dim=-1)  # (B, K, d_in)
        else:
            X = H

        # pad 位置置零
        if sent_mask is not None:
            X = X * sent_mask.unsqueeze(-1)

        # RNN / LSTM / GRU
        if self.backbone_type in ["rnn", "lstm", "gru"]:
            X, _ = self.backbone(X)  # (B, K, d_backbone_out)

        # CNN: Stage1 + Stage2 都在句子轴 K 上卷积
        elif self.backbone_type == "cnn":
            # (B, K, d_in) -> (B, d_in, K)
            X_c = X.transpose(1, 2)

            # Stage 1: MultiScale CNN
            X_c = self.cnn_stage1(X_c)     # (B, c1, K)

            # Stage 2: 再卷一层
            X_c = self.cnn_stage2(X_c)     # (B, c1, K)

            # 回到 (B, K, c1)
            X = X_c.transpose(1, 2)

        # backbone == "none": 直接用 X

        Z = self.ln(X)
        Z = self.dropout(Z)
        logits = self.head(Z).squeeze(-1)  # (B, K)

        return logits
