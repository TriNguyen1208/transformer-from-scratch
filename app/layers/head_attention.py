import torch
import torch.nn as nn
from torch.nn import functional as F
from app.config.constant import EMBED_DIM, HEAD_SIZE, MAX_SEQ_LEN, DROP_OUT
import math

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.W_query = nn.Linear(EMBED_DIM, HEAD_SIZE)
        self.W_key = nn.Linear(EMBED_DIM, HEAD_SIZE)
        self.W_value = nn.Linear(EMBED_DIM, HEAD_SIZE)

        self.register_buffer('tril', torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)))
        self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, input, mask):
        k = self.W_key(input) #(batch_size, seq_len, head_size)
        q = self.W_query(input) #(batch_size, seq_len, head_size)
        v = self.W_value(input) #(batch_size, seq_len, head_size)

        #cong thuc tinh score
        score = q @ k.transpose(-2, -1) / (HEAD_SIZE ** -0.5)

        #If the value equals to 0, set it to -INF.
        score = score.masked_fill(self.tril[:MAX_SEQ_LEN, :MAX_SEQ_LEN] == 0, -math.inf)

        if mask is not None:
            score = score.masked_fill(mask == True, -math.inf)

        score = F.softmax(score, dim=-1)
        score = self.dropout(score) #10% bo di gia tri cua softmax

        output = score @ v
        return output #shape(batch_size, seq_len, head_size)