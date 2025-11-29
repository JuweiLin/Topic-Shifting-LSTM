TRAIN_PATH = "dataset/train.json"
DEV_PATH = "dataset/validation.json"
TEST_PATH = "dataset/test.json"

ENCODER_NAME = "FacebookAI/roberta-large"
UNFREEZE_TOP = 8

BACKBONE = "cnn" # "none" / "rnn" / "lstm" / "gru" / "cnn"
FEATURE_SET = [
    "roles",               # 原来的角色 embedding
    "sent_len",            # 原来的句长
    "position",            # 原来的句子位置
    "speaker_change",      # 1：说话人是否变化
    "run_len_same_speaker",# 1：同一说话人连续长度
    "is_question",         # 2：当前句是否问句
    "disc_marker",         # 2：是否包含显式话题 marker
]

BATCH_SIZE_DIALOG = 8
LR_ENC = 1e-5
LR_HEAD = 1e-5
WEIGHT_DECAY = 0.01
EPOCHS = 20
SEED = 42

LOSS_TYPE = "focal"
GRAD_ACCUM_STEPS = 1
#------- BCE Loss--------
POS_WEIGHT = 1.5
#-------Focal Loss--------
FOCAL_ALPHA = 0.25      # 正类权重，常见 0.25 / 0.5
FOCAL_GAMMA = 2.0       # focusing 参数，常见 2.0
#-------CNN Config--------
CNN_FILTERS = 512
CNN_KERNEL_SIZES = [2, 3]

DROPOUT = 0.5

NUM_WORKERS = 0
