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
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYER)])
        self.layer_norm = nn.LayerNorm(EMBED_DIM)
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, input, target=None):
        token_embed = self.token_embedding(input)
        x = token_embed + self.position_encoding()
        
        pad_mask = (input == GLOBAL_TOKENIZER.PAD_ID).unsqueeze(2)
        pad_mask = pad_mask & pad_mask.transpose(1, 2)
        
        for block in self.blocks:
            x = block(x, pad_mask)

        x = self.layer_norm(x)                          # [B, S, E]
        logits = self.fc_out(x)                         # [B, S, V]

        # negative_infinity = torch.full_like(logits[..., GLOBAL_TOKENIZER.PAD_ID], -float('inf'))
        # logits = logits.scatter_(-1, torch.tensor([GLOBAL_TOKENIZER.PAD_ID], device=logits.device).expand_as(logits[..., GLOBAL_TOKENIZER.PAD_ID].unsqueeze(-1)), 
        #                         negative_infinity.unsqueeze(-1))

        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=GLOBAL_TOKENIZER.PAD_ID)
            return logits, loss

        return logits, None

from app.config.constant import EMBED_DIM, NUM_HEAD, NUM_LAYER

class LibTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(EMBED_DIM, NUM_HEAD, 4 * EMBED_DIM, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYER)
        self.fc = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x, target=None):
        x = self.embedding(x)
        x = self.transformer(x)

        logits = self.fc(x)

        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=GLOBAL_TOKENIZER.PAD_ID)
            return logits, loss

        return logits, None