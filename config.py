TRAIN_PATH = "dataset/train.json"
DEV_PATH = "dataset/validation.json"
TEST_PATH = "dataset/test.json"

ENCODER_NAME = "FacebookAI/roberta-base"
UNFREEZE_TOP = 8

BACKBONE = "rnn"
FEATURE_SET = "none"

BATCH_SIZE_DIALOG = 8
LR_ENC = 1e-5
LR_HEAD = 2e-4
WEIGHT_DECAY = 0.01
EPOCHS = 9
SEED = 42

LOSS_TYPE = "bce"
GRAD_ACCUM_STEPS = 1

POS_WEIGHT = 2.5

FOCAL_ALPHA = 0.25      # 正类权重，常见 0.25 / 0.5
FOCAL_GAMMA = 2.0       # focusing 参数，常见 2.0

NUM_WORKERS = 0
