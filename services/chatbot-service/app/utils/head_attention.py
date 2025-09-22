import torch.nn as nn
from app.config.constant import embed_dim, head_size, seq_len, drop_out
import torch
from torch.nn import functional as F
import math
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_query = nn.Linear(embed_dim, head_size)
        self.W_key = nn.Linear(embed_dim, head_size)
        self.W_value = nn.Linear(embed_dim, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len))) # Because when studying, only study previous thing
        self.dropout = nn.Dropout(drop_out)

    def forward(self, input):
        k = self.W_key(input) #(batch_size, seq_len, head_size)
        q = self.W_query(input) #(batch_size, seq_len, head_size)
        v = self.W_value(input) #(batch_size, seq_len, head_size)

        #cong thuc tinh score
        score = q @ k.transpose(-2, -1) / (head_size ** -0.5)
        #nhung gia tri nao ma da loc = 0 thi cho la -inf. De khi vao softmax thi no la 0
        score = score.masked_fill(self.tril[:seq_len, :seq_len] == 0, -math.inf)
        score = F.softmax(score, dim=-1)
        score = self.dropout(score) #10% bo di gia tri cua softmax

        output = score @ v
        return output