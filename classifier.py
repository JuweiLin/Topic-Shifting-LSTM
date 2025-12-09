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
            self.bns.append(nn.Identity())
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C_in, K)
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            o = conv(x)
            o = bn(o)
            o = F.relu(o)
            o = self.dropout(o)
            outs.append(o)
        return torch.cat(outs, dim=1)


    def forward(self, x):
        # x: (B, C_in, K)
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            o = conv(x)       # (B, F, K)
            o = bn(o)
            o = F.relu(o)
            o = self.dropout(o)
            outs.append(o)
        return torch.cat(outs, dim=1)


    def forward(self, x):
        # x: (B, C_in, K)
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            o = conv(x)       # (B, F, K)
            o = bn(o)
            o = F.relu(o)
            o = self.dropout(o)
            outs.append(o)
        return torch.cat(outs, dim=1)


class SentenceShiftClassifier(nn.Module):
    def __init__(
        self,
        d_enc,
        d_feat=0,
        backbone="lstm", # "none" / "rnn" / "lstm" / "gru" / "cnn"
        hidden_size=512,
        num_layers=2,
        dropout=0.2,
        cnn_filters=200,
        cnn_kernel_sizes=(2,3,4),
        head_hidden=256,
        **kwargs,
    ):
        super().__init__()
        self.backbone_type = backbone
        d_in = d_enc + d_feat

        self.backbone = None
        self.cnn_stage1 = None
        self.cnn_stage2 = None
        d_backbone_out = d_in

        if backbone == "none":
            self.backbone = None
            d_backbone_out = d_in

        #RNN / LSTM / GRU
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

        elif backbone == "cnn":
            self.cnn_stage1 = MultiScaleCNN(
                input_dim=d_in,
                num_filters=cnn_filters,
                kernel_sizes=cnn_kernel_sizes,
                dropout=dropout,
            )
            c1 = cnn_filters * len(cnn_kernel_sizes)

            self.cnn_stage2 = nn.Sequential(
                nn.Conv1d(c1, c1, kernel_size=3, padding=1),
                nn.BatchNorm1d(c1),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            d_backbone_out = c1

        else:
            raise ValueError("Unknown backbone: %s" % backbone)

        # Norm + Dropout + MLP head
        self.ln = nn.LayerNorm(d_backbone_out)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_backbone_out, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, H, sent_mask, F=None):
        if F is not None:
            X = torch.cat([H, F], dim=-1)  # (B, K, d_in)
        else:
            X = H

        if sent_mask is not None:
            X = X * sent_mask.unsqueeze(-1)

        # RNN / LSTM / GRU
        if self.backbone_type in ["rnn", "lstm", "gru"]:
            X, _ = self.backbone(X)  # (B, K, d_backbone_out)

        # CNN: Stage1 + Stage2 
        elif self.backbone_type == "cnn":
            # (B, K, d_in) -> (B, d_in, K)
            X_c = X.transpose(1, 2)

            # Stage 1: MultiScale CNN
            X_c = self.cnn_stage1(X_c)     # (B, c1, K)

            # Stage 2: Conv + BN + ReLU + Dropout
            X_c = self.cnn_stage2(X_c)     # (B, c1, K)

            # (B, K, c1)
            X = X_c.transpose(1, 2)


        Z = self.ln(X)
        Z = self.dropout(Z)
        logits = self.head(Z).squeeze(-1)  # (B, K)

        return logits
