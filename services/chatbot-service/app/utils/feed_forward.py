import torch.nn as nn
from app.config.constant import embed_dim, drop_out
from torch.nn import functional as F
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(drop_out)
        )
    def forward(self, input):
        return self.net(input)