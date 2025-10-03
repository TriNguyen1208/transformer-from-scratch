import torch.nn as nn
from app.layers.multihead_attention import MultiHeadAttention
from app.layers.feed_forward import FeedForward
from app.config.constant import EMBED_DIM

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.multi_head = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(EMBED_DIM)
        self.layer_norm_2 = nn.LayerNorm(EMBED_DIM)

    def forward(self, input, mask=None):
        input = input + self.multi_head(input, mask)
        self.layer_norm_1(input)

        input = input + self.feed_forward(input)
        self.layer_norm_2(input)
        
        return input