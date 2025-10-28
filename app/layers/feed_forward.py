import torch.nn as nn
from app.config.constant import EMBED_DIM, DROP_OUT

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.GELU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
            nn.Dropout(DROP_OUT)
        )
        
    def forward(self, input):
        return self.net(input)