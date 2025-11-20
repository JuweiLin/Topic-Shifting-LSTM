# feature.py

import torch
import torch.nn as nn


class SentenceFeatureBuilder(nn.Module):
    """
    feature_set:
      - "none" 或 ""  -> 不用任何句级特征
      - 列表或逗号分隔字符串:
        例如 ["roles", "sent_len", "position", "is_question"]
        或   "roles,sent_len,position,is_question"

    现在内置支持的 key（同时也是 batch/meta 里的字段名）：
      - "roles"          : 说话人角色 id，做 embedding
      - "sent_len"       : 句长（log 之后归一化） -> 1 维
      - "position"       : 在对话中的位置归一化   -> 1 维
      - "is_question"    : 是否问句          -> 1 维
      - "has_marker"     : 是否有话题标记    -> 1 维
      - "speaker_switch" : 是否换说话人      -> 1 维
      - "time_shift"     : 是否时间/场景跳转  -> 1 维
    """

    def __init__(self, feature_set="none"):
        super().__init__()

        # ---- 解析 feature_set 成 list ----
        if isinstance(feature_set, str):
            s = feature_set.strip()
            if s == "" or s == "none":
                feature_list = []
            else:
                feature_list = [x.strip() for x in s.split(",") if x.strip()]
        else:
            feature_list = list(feature_set) if feature_set is not None else []

        self.feature_list = feature_list

        if len(self.feature_list) == 0:
            self.d_feat = 0
            return

        in_dim = 0

        # roles: embedding
        self.use_roles = "roles" in self.feature_list
        if self.use_roles:
            self.role_dim = 8
            self.num_roles = 4
            self.role_emb = nn.Embedding(self.num_roles, self.role_dim)
            in_dim += self.role_dim

        # 标量类 feature，都是 1 维
        self.use_sent_len       = "sent_len"       in self.feature_list
        self.use_position       = "position"       in self.feature_list
        self.use_is_question    = "is_question"    in self.feature_list
        self.use_has_marker     = "has_marker"     in self.feature_list
        self.use_speaker_switch = "speaker_switch" in self.feature_list
        self.use_time_shift     = "time_shift"     in self.feature_list

        for flag in [
            self.use_sent_len,
            self.use_position,
            self.use_is_question,
            self.use_has_marker,
            self.use_speaker_switch,
            self.use_time_shift,
        ]:
            if flag:
                in_dim += 1

        # 统一映射到一个比较大的维度
        self.d_feat = 64
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2 * self.d_feat),
            nn.ReLU(),
            nn.Linear(2 * self.d_feat, self.d_feat),
        )

    def get_dim(self):
        return self.d_feat

    # =============== 每个 feature 一个函数 ===============

    def _feat_roles(self, meta):
        roles = meta["roles"]                # (B, K)
        roles = roles.clamp(min=0, max=self.num_roles - 1)
        return self.role_emb(roles)         # (B, K, role_dim)

    def _feat_sent_len(self, meta):
        sent_len = meta["sent_len"].float()  # (B, K)
        norm_len = torch.log1p(sent_len) / 5.0
        return norm_len.unsqueeze(-1)        # (B, K, 1)

    def _feat_position(self, meta):
        position = meta["position"].float()  # (B, K)
        norm_pos = (position / 100.0).unsqueeze(-1)
        return norm_pos                      # (B, K, 1)

    def _feat_is_question(self, meta):
        if "is_question" in meta:
            v = meta["is_question"].float()
        else:
            v = torch.zeros_like(meta["sent_len"], dtype=torch.float32)
        return v.unsqueeze(-1)

    def _feat_has_marker(self, meta):
        if "has_marker" in meta:
            v = meta["has_marker"].float()
        else:
            v = torch.zeros_like(meta["sent_len"], dtype=torch.float32)
        return v.unsqueeze(-1)

    def _feat_speaker_switch(self, meta):
        if "speaker_switch" in meta:
            v = meta["speaker_switch"].float()
        else:
            v = torch.zeros_like(meta["sent_len"], dtype=torch.float32)
        return v.unsqueeze(-1)

    def _feat_time_shift(self, meta):
        if "time_shift" in meta:
            v = meta["time_shift"].float()
        else:
            v = torch.zeros_like(meta["sent_len"], dtype=torch.float32)
        return v.unsqueeze(-1)

    # =============== forward ===============

    def forward(self, batch_meta, sent_mask):
        if self.d_feat == 0 or len(self.feature_list) == 0:
            return None

        feat_chunks = []

        # 按 feature_list 顺序依次调对应函数
        for name in self.feature_list:
            func_name = f"_feat_{name}"
            if not hasattr(self, func_name):
                raise ValueError(
                    f"Feature '{name}' is in FEATURE_SET but method '{func_name}' is not implemented."
                )
            Fi = getattr(self, func_name)(batch_meta)  # (B, K, d_i)
            feat_chunks.append(Fi)

        feat_cat = torch.cat(feat_chunks, dim=-1)      # (B, K, in_dim)
        F = self.mlp(feat_cat)                         # (B, K, d_feat)
        F = F * sent_mask.unsqueeze(-1)                # mask padding

        return F