import torch.nn as nn
import torch
from app.config.constant import embed_dim, head_size, num_head, drop_out
from app.utils.head_attention import Head
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        assert embed_dim % num_head == 0
        
        #duyet qua num_head lan Head
        self.heads = nn.ModuleList([Head() for _ in range(num_head)])
        self.proj = nn.Linear(head_size * num_head, embed_dim) #Buoc LinearProjection la chuyen shape (head_size * num_head) sang embed_dim
        self.dropout = nn.Dropout(drop_out) #co 10% la bo bot di 1 so neuron
    def forward(self, input):
        out = torch.cat([h(input) for h in self.heads], dim=-1) #out co shape la (batch_size, seq_len, head_size * num_head)
        #Note: con thieu buoc dropout.
        out = self.dropout(self.proj(out))
        return out #shape luc nay la (batch_size, seq_len, embed_dim)