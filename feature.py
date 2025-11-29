# feature.py

import torch
import torch.nn as nn
from tables import ROLE2ID


class SentenceFeatureBuilder(nn.Module):
    """
    feature_set:
      - "none" 或 ""  -> 不用任何句级特征
      - 列表或逗号分隔字符串:
        例如 ["roles", "sent_len", "position", "is_question"]
        或   "roles,sent_len,position,is_question"

    支持的 key（也是 batch/meta 里的字段名）：
      - "roles"
      - "sent_len"
      - "position"
      - "speaker_change"
      - "run_len_same_speaker"
      - "is_question"
      - "disc_marker"
    """

    def __init__(self, feature_set="none", d_feat=64):
        super().__init__()

        # 解析 feature_set
        if isinstance(feature_set, str):
            s = feature_set.strip()
            if s == "" or s == "none":
                feature_list = []
            elif s == "basic":
                feature_list = ["roles", "sent_len", "position"]
            else:
                feature_list = [x.strip() for x in s.split(",") if x.strip()]
        else:
            feature_list = list(feature_set) if feature_set is not None else []

        self.feature_list = feature_list

        # 没有特征就直接关掉
        if len(self.feature_list) == 0:
            self.d_feat = 0
            self.mlp = None
            return

        # roles embedding 维度
        self.use_roles = "roles" in self.feature_list
        in_dim = 0

        if self.use_roles:
            self.num_roles = len(ROLE2ID)   # 根据 TIAGE 的角色字典自动确定
            self.role_dim = 8
            self.role_emb = nn.Embedding(self.num_roles, self.role_dim)
            in_dim += self.role_dim

        # 标量类 feature
        self.use_sent_len            = "sent_len"            in self.feature_list
        self.use_position            = "position"            in self.feature_list
        self.use_speaker_change      = "speaker_change"      in self.feature_list
        self.use_run_len_same_speaker= "run_len_same_speaker"in self.feature_list
        self.use_is_question         = "is_question"         in self.feature_list
        self.use_disc_marker         = "disc_marker"         in self.feature_list

        for flag in [
            self.use_sent_len,
            self.use_position,
            self.use_speaker_change,
            self.use_run_len_same_speaker,
            self.use_is_question,
            self.use_disc_marker,
        ]:
            if flag:
                in_dim += 1

        self.d_feat = d_feat
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2 * self.d_feat),
            nn.ReLU(),
            nn.Linear(2 * self.d_feat, self.d_feat),
        )

    def get_dim(self):
        return self.d_feat

    # ====== 每个 feature 一个小函数 ======

    def _feat_roles(self, meta):
        roles = meta["roles"]                   # (B, K)
        roles = roles.clamp(min=0, max=self.num_roles - 1)
        return self.role_emb(roles)            # (B, K, role_dim)

    def _feat_sent_len(self, meta):
        sent_len = meta["sent_len"].float()     # (B, K)
        norm_len = torch.log1p(sent_len) / 5.0
        return norm_len.unsqueeze(-1)           # (B, K, 1)

    def _feat_position(self, meta):
        position = meta["position"].float()     # (B, K)
        norm_pos = (position / 100.0).unsqueeze(-1)
        return norm_pos                         # (B, K, 1)

    def _feat_speaker_change(self, meta):
        v = meta.get("speaker_change", torch.zeros_like(meta["sent_len"])).float()
        return v.unsqueeze(-1)

    def _feat_run_len_same_speaker(self, meta):
        v = meta.get("run_len_same_speaker", torch.zeros_like(meta["sent_len"])).float()
        v = torch.log1p(v) / 5.0
        return v.unsqueeze(-1)

    def _feat_is_question(self, meta):
        v = meta.get("is_question", torch.zeros_like(meta["sent_len"])).float()
        return v.unsqueeze(-1)

    def _feat_disc_marker(self, meta):
        v = meta.get("disc_marker", torch.zeros_like(meta["sent_len"])).float()
        return v.unsqueeze(-1)

    # ====== forward ======

    def forward(self, batch_meta, sent_mask):
        """
        batch_meta: dict，包含 roles/sent_len/position 以及选中的其它特征
        sent_mask: (B, K)
        """
        if self.d_feat == 0 or self.mlp is None or len(self.feature_list) == 0:
            return None

        feat_chunks = []

        for name in self.feature_list:
            func_name = f"_feat_{name}"
            if not hasattr(self, func_name):
                raise ValueError(
                    f"Feature '{name}' is in FEATURE_SET but '{func_name}' is not implemented in SentenceFeatureBuilder."
                )
            Fi = getattr(self, func_name)(batch_meta)   # (B, K, d_i)
            feat_chunks.append(Fi)

        feat_cat = torch.cat(feat_chunks, dim=-1)       # (B, K, in_dim)
        F = self.mlp(feat_cat)                          # (B, K, d_feat)
        F = F * sent_mask.unsqueeze(-1)

        return F
