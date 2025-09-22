from app.utils.processor import Preprocessor
from app.utils.tokenizer import Tokenizer
import os
from app.models.transformer import TransformerModel

#constant variable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_NAME = 'dataset.txt'
embed_dim = 128







# tokenizer = Tokenizer()
# tokenizer.buildVocabs(data_txt)

# # vocab_size = len(tokenizer.word2idx)
# # model = Transformer(vocab_size)

# sentence = 'leeh xin chào bạn'
# ids = tokenizer.encode(sentence)
# print(ids)

# print('=' * 10)

# decoding = tokenizer.decode(ids)
# print(decoding)