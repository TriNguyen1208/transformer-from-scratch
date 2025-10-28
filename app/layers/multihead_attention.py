import torch.nn as nn
import torch
from app.config.constant import EMBED_DIM, HEAD_SIZE, NUM_HEAD, DROP_OUT
from app.layers.head_attention import Head

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        assert EMBED_DIM % NUM_HEAD == 0
        
        #duyet qua (NUM_HEAD) lan Head
        self.heads = nn.ModuleList([Head() for _ in range(NUM_HEAD)])
        self.proj = nn.Linear(HEAD_SIZE * NUM_HEAD, EMBED_DIM) #Buoc LinearProjection la chuyen shape (head_size * num_head) sang embed_dim
        self.dropout = nn.Dropout(DROP_OUT) #co 10% la bo bot di 1 so neuron

    def forward(self, input, mask=None):
        out = torch.cat([h(input, mask) for h in self.heads], dim=-1) #out co shape la (batch_size, seq_len, head_size * num_head)
        
        out = self.dropout(self.proj(out))

        return out #shape luc nay la (batch_size, seq_len, embed_dim)