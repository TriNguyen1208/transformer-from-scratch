import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=128, max_seq_len=1000):
        super().__init__()

        pe_list = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe_list[:, 0::2] = torch.sin(position * div_term)
        pe_list[:, 1::2] = torch.cos(position * div_term)

        pe_list = pe_list.unsqueeze(0)
        self.register_buffer('pe', pe_list)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        
        assert embed_dim % n_head == 0

        self.embed_dim = embed_dim
        self.nhead = n_head
        self.ed_nhead = self.embed_dim // self.nhead
        
        self.W_query = nn.Linear(embed_dim, embed_dim)
        self.W_key = nn.Linear(embed_dim, embed_dim)
        self.W_value = nn.Linear(embed_dim, embed_dim)
        self.W_output = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_inp, key_inp, value_inp, mask=None):
        batch_size, seq_len, _ = query_inp.size()

        query = self.W_query(query_inp).view(batch_size, seq_len, self.nhead, self.ed_nhead).transpose(1, 2)
        key = self.W_key(key_inp).view(batch_size, seq_len, self.nhead, self.ed_nhead).transpose(1, 2)
        value = self.W_value(value_inp).view(batch_size, seq_len, self.nhead, self.ed_nhead).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.ed_nhead)
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, value)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.W_output(out)