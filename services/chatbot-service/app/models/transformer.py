import torch
import torch.nn as nn
import math
from app.utils.tokenizer import Tokenizer
from app.utils.position_encoding import PositionalEncoding
from app.config.constant import embed_dim, seq_len, vocab_size, num_layer
from app.utils.utils import get_batch
from app.utils.block import TransformerBlock
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEncoding(embed_dim=embed_dim, max_seq_len=seq_len)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input, target=None):
        token_embed = self.token_embedding(input)
        x = token_embed + self.position_embedding()
        x = self.blocks(x)
        #toi day lam gi nua ta:)))
            
