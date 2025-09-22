from app.utils.processor import Preprocessor
from app.utils.tokenizer import Tokenizer
import os
from app.config.constant import BASE_DIR, FILE_NAME, device, max_iter, learning_rate
from app.models.transformer import TransformerModel
from app.utils.utils import get_batch
import torch
#file handler
dataset_path = os.path.join(BASE_DIR, FILE_NAME)

#Load file
text = Preprocessor.loadText([dataset_path]) #Load text

#Build the vocab
tokenizer = Tokenizer()
tokenizer.buildVocabs(text)

#Calculate vocab size
vocab_size = len(tokenizer.word2idx)

#encode text
text_encode = torch.tensor(tokenizer.encode(text), dtype=torch.long)

model = TransformerModel()
model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for _ in range(2):
    x_batch, y_batch = get_batch(text_encode)
    model(x_batch, y_batch)
