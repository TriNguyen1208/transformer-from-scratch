import torch
import torch.nn as nn
import math
from app.config.constant import EMBED_DIM, MAX_SEQ_LEN

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

        pe_list = torch.zeros(MAX_SEQ_LEN, EMBED_DIM)
        position = torch.arange(0, MAX_SEQ_LEN, dtype=torch.float).reshape(-1, 1)
        div_term = torch.exp(torch.arange(0, EMBED_DIM, 2).float() * (-math.log(10000.0) / EMBED_DIM))
        
        pe_list[:, 0::2] = torch.sin(position * div_term)
        pe_list[:, 1::2] = torch.cos(position * div_term)

        pe_list = pe_list.reshape(1, MAX_SEQ_LEN, EMBED_DIM) # shape (1, max_len, embed_dim)
        self.register_buffer('pe', pe_list)

    def forward(self):
        return self.pe