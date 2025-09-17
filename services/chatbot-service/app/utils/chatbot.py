from processor import Preprocessor
from tokenizer import Tokenizer
from model import Transformer
import torch
import math

# data_txt = Preprocessor.loadText(['./data/dataset.txt'])

# tokenizer = Tokenizer()
# tokenizer.buildVocabs(data_txt)

# vocab_size = len(tokenizer.word2idx)
# model = Transformer(vocab_size)

# sentence = 'chia đều cho tất cả chúng'
# ids = tokenizer.encode(sentence)

# x = torch.tensor([ids])

# with torch.no_grad():
#     output = model(x)

# last_token_logits = output[0, -1, :]

# pred_id = torch.argmax(last_token_logits).item()
# pred_word = tokenizer.idx2word[pred_id]

# print("Predicted next word:", pred_word)

d_model = 16
max_len = 10

div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
pe = torch.zeros(max_len, d_model)


print(pe[:, 0::2])