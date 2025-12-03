TRAIN_PATH = "dataset/train.json"
DEV_PATH = "dataset/validation.json"
TEST_PATH = "dataset/test.json"

ENCODER_NAME = "FacebookAI/roberta-large"
UNFREEZE_TOP = -1  # 0: freeze all transformer blocks; -1: unfreeze all; >0: unfreeze last N layers

BACKBONE = "cnn"  # "none" / "rnn" / "lstm" / "gru" / "cnn"

FEATURE_SET = [
    "roles",               # 角色 embedding
    "sent_len",            # 句长
    "position",            # 句子位置
    "speaker_change",      # 说话人是否变化
    "run_len_same_speaker",# 同一说话人连续长度
    "is_question",         # 当前句是否问句
    "disc_marker",         # 是否包含显式话题 marker
]

BATCH_SIZE_DIALOG = 8

LR_ENC = 1e-5   # encoder 学习率
LR_HEAD = 1e-5  # 分类头和特征层学习率
WEIGHT_DECAY = 0.01
EPOCHS = 20
SEED = 77

LOSS_TYPE = "focal"   # "bce" / "focal"
GRAD_ACCUM_STEPS = 1

# ------- BCE Loss --------
POS_WEIGHT = 1.5

# ------- Focal Loss -------
FOCAL_ALPHA = 0.1
FOCAL_GAMMA = 2.0

# ------- CNN Config -------
CNN_FILTERS = 512
CNN_KERNEL_SIZES = [2, 3]

DROPOUT = 0.3

NUM_WORKERS = 0

# special sentence delimiter token，后面 tokenizer 会保证把它加入 vocab
SENT_TOKEN = "<sent>"
