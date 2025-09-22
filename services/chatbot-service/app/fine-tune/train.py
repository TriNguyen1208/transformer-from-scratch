from app.utils.processor import Preprocessor
from app.utils.tokenizer import Tokenizer
import os
from app.config.constant import BASE_DIR, FILE_NAME, device, max_iter, learning_rate, text_encode
from app.models.transformer import TransformerModel
from app.utils.utils import get_batch
import torch


model = TransformerModel()
model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for _ in range(2):
    x_batch, y_batch = get_batch(text_encode)
    model(x_batch, y_batch)
