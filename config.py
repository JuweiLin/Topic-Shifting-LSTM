TRAIN_PATH = "dataset/train.json"
DEV_PATH = "dataset/validation.json"
TEST_PATH = "dataset/test.json"

ENCODER_NAME = "FacebookAI/roberta-large"
UNFREEZE_TOP = -1  # 0: freeze all transformer blocks; -1: unfreeze all; >0: unfreeze last N layers

BACKBONE = "cnn"  # "none" / "rnn" / "lstm" / "gru" / "cnn"

FEATURE_SET = [
    "roles",
    "sent_len",
    "position",
    "speaker_change",
    "run_len_same_speaker",
    "is_question",
    "disc_marker",
]

BATCH_SIZE_DIALOG = 8

LR_ENC = 1e-5
LR_HEAD = 1e-5
WEIGHT_DECAY = 0.05
EPOCHS = 20
SEED = 43

LOSS_TYPE = "focal"   # "bce" / "focal"
GRAD_ACCUM_STEPS = 1

# ------- BCE Loss --------
POS_WEIGHT = 1.5

# ------- Focal Loss -------
FOCAL_ALPHA = None
FOCAL_GAMMA = 2.0

# ------- CNN Config -------
CNN_FILTERS = 512
CNN_KERNEL_SIZES = [2, 3]

DROPOUT = 0.3

NUM_WORKERS = 0

# special sentence delimiter token
SENT_TOKEN = "<sent>"
