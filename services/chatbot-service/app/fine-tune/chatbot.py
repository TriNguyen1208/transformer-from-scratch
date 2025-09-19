from utils.processor import Preprocessor
from utils.tokenizer import Tokenizer
# from ..models.model import Transformer

data_txt = Preprocessor.loadText(['../dataset.txt'])

tokenizer = Tokenizer()
tokenizer.buildVocabs(data_txt)

# vocab_size = len(tokenizer.word2idx)
# model = Transformer(vocab_size)

sentence = 'leeh xin chào bạn'
ids = tokenizer.encode(sentence)
print(ids)

print('=' * 10)

decoding = tokenizer.decode(ids)
print(decoding)