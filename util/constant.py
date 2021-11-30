from enum import Enum

WINDOW_SIZE = 5
FEATURE_NUMS = 5
EPOCHS = 100
LEARNING_RATE = 0.008
WEIGHT_DECAY = 1e-3
EPSILON = 1e-10
MAX_ACC = 0.0
DATA_PATH = r'F:\project\big_work\dataset\南昌2020气温数据.csv'


class Env(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3
    ALL = 4
