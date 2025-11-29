TRAIN_PATH = "dataset/train.json"
DEV_PATH = "dataset/validation.json"
TEST_PATH = "dataset/test.json"

ENCODER_NAME = "FacebookAI/roberta-large"
UNFREEZE_TOP = 8

BACKBONE = "cnn" # "none" / "rnn" / "lstm" / "gru" / "cnn"
FEATURE_SET = [
    # "roles",
    # "sent_len",
    # "position",
    # "is_question",
    # "has_marker",
    # "speaker_switch",
    # "time_shift",
]

BATCH_SIZE_DIALOG = 8
LR_ENC = 1e-5
LR_HEAD = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 9
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
