import os
import torch.nn as nn
import torch
from app.utils.tokenizer import Tokenizer
from app.utils.processor import Preprocessor

#constant variable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_NAME = 'dataset.txt'
embed_dim = 128
batch_size = 64
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
seq_len = 50
batch_size = 10
max_iter = 5000
num_head = 4
head_size = embed_dim // num_head
drop_out = 0.1
num_layer = 4
#file handler
dataset_path = os.path.join(BASE_DIR, FILE_NAME)

text = Preprocessor.loadText([dataset_path]) #Load text
#Build the vocab
tokenizer = Tokenizer()
tokenizer.buildVocabs(text)

#Calculate vocab size
vocab_size = len(tokenizer.word2idx)

#encode text
text_encode = torch.tensor(tokenizer.encode(text), dtype=torch.long)
