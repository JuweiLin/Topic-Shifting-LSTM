# tables.py
# user / agent
ROLE2ID = {
    "user": 0,
    "agent": 1,
    "other": 2,
}
ID2ROLE = {v: k for k, v in ROLE2ID.items()}

# segmentation_label
SEG_LABEL_ID2NAME = {
    0: "no_shift",
    1: "shift",
}

# disc_marker
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
