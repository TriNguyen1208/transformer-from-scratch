import torch
import torch.nn as nn
from torch.nn import functional as F
from app.config.constant import EMBED_DIM, NUM_LAYER
from app.config.config import GLOBAL_TOKENIZER
from app.layers.position_encoding import PositionalEncoding
from app.layers.block import TransformerBlock

vocab_size = GLOBAL_TOKENIZER.getVocabSize()

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_encoding = PositionalEncoding()
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYER)])
        self.layer_norm = nn.LayerNorm(EMBED_DIM)
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, input, target=None):
        token_embed = self.token_embedding(input)
        x = token_embed + self.position_encoding()
        x = self.blocks(x)

        x = self.layer_norm(x)                          # [B, S, E]
        logits = self.fc_out(x)                         # [B, S, V]

        pad_index = GLOBAL_TOKENIZER.PAD_ID

        negative_infinity = torch.full_like(logits[..., pad_index], -float('inf'))

        logits = logits.scatter_(-1, 
                                torch.tensor([pad_index], device=logits.device).expand_as(logits[..., pad_index].unsqueeze(-1)), 
                                negative_infinity.unsqueeze(-1))

        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            return logits, loss

        return logits, None