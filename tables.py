# tables.py
"""
集中放 TIAGE 数据集的映射表和词表。
"""

# TIAGE 的说话人角色：user / agent
ROLE2ID = {
    "user": 0,
    "agent": 1,
    "other": 2,   # 兜底用
}
ID2ROLE = {v: k for k, v in ROLE2ID.items()}

# segmentation_label 在 TIAGE 本身就是 0 / 1，这里只做个名字说明
SEG_LABEL_ID2NAME = {
    0: "no_shift",
    1: "shift",
}

# 显式话题标记，用来算 disc_marker
DISCOURSE_MARKERS = [
    "anyway",
    "by the way",
    "btw",
    "now",
    "okay",
    "ok",
    "so ",
    "so,",
    "well ",
    "well,",
]
