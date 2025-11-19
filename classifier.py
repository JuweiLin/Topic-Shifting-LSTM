# classifier.py

import torch
import torch.nn as nn


class SentenceShiftClassifier(nn.Module):
    def __init__(
        self,
        d_enc,
        d_feat=0,
        backbone="lstm",    # "none" / "rnn" / "lstm" / "gru"
        hidden_size=512,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.backbone_type = backbone
        d_in = d_enc + d_feat

        self.backbone = None
        d_backbone_out = d_in

        if backbone == "none":
            self.backbone = None
            d_backbone_out = d_in
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
        else:
            raise ValueError("Unknown backbone: %s" % backbone)

        self.ln = nn.LayerNorm(d_backbone_out)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_backbone_out, 1)

    def forward(self, H, sent_mask, F=None):
        if F is not None:
            X = torch.cat([H, F], dim=-1)  # (B, K, d_in)
        else:
            X = H

        if self.backbone is not None:
            X, _ = self.backbone(X)  # (B, K, d_backbone_out)

        Z = self.ln(X)
        Z = self.dropout(Z)
        logits = self.head(Z).squeeze(-1)  # (B, K)

        return logits
