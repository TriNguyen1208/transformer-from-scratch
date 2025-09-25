import torch
import os

#constant variable
EMBED_DIM = 256
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_SEQ_LEN = 32
BATCH_SIZE = 128
MAX_ITER = 50
NUM_HEAD = 4
HEAD_SIZE = EMBED_DIM // NUM_HEAD
DROP_OUT = 0.1
NUM_LAYER = 4
NUM_EPOCHS = 5

#file handler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FILE_NAME = r"data\dataset.txt"
CHECKPOINT_FILE_NAME = r"models\model.pkl"

DATASET_PATH = os.path.join(BASE_DIR, DATASET_FILE_NAME)
CHECKPOINT_PATH = os.path.join(BASE_DIR, CHECKPOINT_FILE_NAME)