# feature.py

import torch
import torch.nn as nn


class SentenceFeatureBuilder(nn.Module):
    def __init__(self, feature_set="none"):
        super().__init__()
        self.feature_set = feature_set

        if feature_set == "basic":
            self.role_dim = 4
            self.num_roles = 4
            self.role_emb = nn.Embedding(self.num_roles, self.role_dim)

            self.d_feat = 16
            self.mlp = nn.Linear(self.role_dim + 2, self.d_feat)
        else:
            self.d_feat = 0

    def get_dim(self):
        return self.d_feat

    def forward(self, batch_meta, sent_mask):
        if self.feature_set == "none":
            return None

        roles = batch_meta["roles"]       # (B, K)
        sent_len = batch_meta["sent_len"] # (B, K)
        position = batch_meta["position"] # (B, K)

        roles_clamped = roles.clamp(min=0, max=self.num_roles - 1)
        role_vec = self.role_emb(roles_clamped)  # (B, K, role_dim)

        sent_len_f = sent_len.float()
        pos_f = position.float()

        norm_len = torch.log1p(sent_len_f) / 5.0
        norm_pos = pos_f / 100.0

        norm_len = norm_len.unsqueeze(-1)  # (B, K, 1)
        norm_pos = norm_pos.unsqueeze(-1)  # (B, K, 1)

        feat_cat = torch.cat([role_vec, norm_len, norm_pos], dim=-1)  # (B, K, role_dim+2)
        F = self.mlp(feat_cat)  # (B, K, d_feat)

        F = F * sent_mask.unsqueeze(-1)

        return F
