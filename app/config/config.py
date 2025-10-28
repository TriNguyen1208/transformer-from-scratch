from app.data.tokenizer import Tokenizer
from app.data.processor import Preprocessor
from app.config.constant import DATASET_PATH

def encodeSentences(sentences):
    encode_text = []
    for sentence in sentences:
        encode_text.append(GLOBAL_TOKENIZER.encode(sentence))

    return encode_text

DATA_TEXT = Preprocessor.loadText([DATASET_PATH])
SENTENCES = Preprocessor.splitIntoSentences(DATA_TEXT)

GLOBAL_TOKENIZER = Tokenizer()
GLOBAL_TOKENIZER.buildVocabs(SENTENCES)

ENCODED_SENTENCES = encodeSentences(SENTENCES)