
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=128, max_seq_len=1000):
        super().__init__()

        pe_list = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).reshape(-1, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe_list[:, 0::2] = torch.sin(position * div_term)
        pe_list[:, 1::2] = torch.cos(position * div_term)

        pe_list = pe_list.reshape(1, max_seq_len, embed_dim) #shape (1, max_len, embed_dim)
        self.register_buffer('pe', pe_list)

    def forward(self):
        return self.pe