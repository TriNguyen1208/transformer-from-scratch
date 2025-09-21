import torch.nn as nn
from app.utils.multihead_attention import MultiHeadAttention
from app.utils.feed_forward import FeedForward
from app.config.constant import embed_dim
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
    def forward(self, input):
        input = input + self.multi_head(input)
        self.layer_norm_1(input)
        input = input + self.feed_forward(input)
        self.layer_norm_2(input)
        return input