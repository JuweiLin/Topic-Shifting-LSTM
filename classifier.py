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
        # x: (B, C, L)
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            o = conv(x)       # (B, F, L)
            o = bn(o)
            o = F.relu(o)
            o = self.dropout(o)
            outs.append(o)
        # 通道维拼接 -> (B, F * len(kernel_sizes), L)
        return torch.cat(outs, dim=1)


class SentenceShiftClassifier(nn.Module):
    def __init__(
        self,
        d_enc,
        d_feat=0,
        backbone="lstm",    # "none" / "rnn" / "lstm" / "gru" / "cnn"
        hidden_size=512,
        num_layers=2,
        dropout=0.2,
        cnn_filters=128,              # 新增: CNN 每个 kernel 的通道数
        cnn_kernel_sizes=(2, 3, 4),   # 新增: 多尺度卷积的 kernel size
        **kwargs,                     # 防止多传别的参数崩掉
    ):
        super().__init__()
        self.backbone_type = backbone
        d_in = d_enc + d_feat

        self.backbone = None
        self.cnn = None
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

        # 3) CNN：这里真正用 MultiScaleCNN
        elif backbone == "cnn":
            self.cnn = MultiScaleCNN(
                input_dim=d_in,
                num_filters=cnn_filters,
                kernel_sizes=cnn_kernel_sizes,
                dropout=dropout,
            )
            # MultiScaleCNN 会在通道维度 concat，输出通道数 = filters * kernel_sizes 个数
            d_backbone_out = cnn_filters * len(cnn_kernel_sizes)

        else:
            raise ValueError("Unknown backbone: %s" % backbone)

        self.ln = nn.LayerNorm(d_backbone_out)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_backbone_out, 1)

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

        # pad 位置置零，干净一点
        if sent_mask is not None:
            X = X * sent_mask.unsqueeze(-1)

        # RNN / LSTM / GRU
        if self.backbone_type in ["rnn", "lstm", "gru"]:
            X, _ = self.backbone(X)  # (B, K, d_backbone_out)

        # CNN：Conv1d 输入是 (B, C, L)，这里 L=句子数K
        elif self.backbone_type == "cnn":
            X_c = X.transpose(1, 2)        # (B, d_in, K)
            X_c = self.cnn(X_c)            # (B, F * len(k), K)
            X = X_c.transpose(1, 2)        # (B, K, F * len(k))

        # backbone == "none" 就直接用 X

        Z = self.ln(X)
        Z = self.dropout(Z)
        logits = self.head(Z).squeeze(-1)  # (B, K)

        return logits
