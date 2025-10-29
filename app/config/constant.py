import torch
import os

#constant variable
EMBED_DIM = 256 # The size elements in the vector that represents each token
LEARNING_RATE = 1e-5 # Learning rate of AdamW optimizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

MAX_SEQ_LEN = 64 # Maximums number of tokens(words) that the model can process in one time. If you over this numbers, cut it.
BATCH_SIZE = 64 # Number of samples proceed in each batch during training
MAX_ITER = 50 #Maximum number of iterations per epoch
NUM_HEAD = 4 #Number of attention head in the multihead attention
HEAD_SIZE = EMBED_DIM // NUM_HEAD # Size of attention head
DROP_OUT = 0.1 #Dropout probabilties - Prevent overfitting by randomly setting neurons to 0
NUM_LAYER = 4 #Numbers of transformers layers
NUM_EPOCHS = 15 #Number of epochs

#file handler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FILE_NAME = r"data\weather-dataset.txt"
CHECKPOINT_FILE_NAME = r"models\weather-model.pt"

DATASET_PATH = os.path.join(BASE_DIR, DATASET_FILE_NAME) #Path of dataset
CHECKPOINT_PATH = os.path.join(BASE_DIR, CHECKPOINT_FILE_NAME) #Path of checkpoints