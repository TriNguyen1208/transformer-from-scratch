from app.data.tokenizer import Tokenizer
from app.data.processor import Preprocessor
from app.config.constant import DATASET_PATH

def encode_sentences(sentences):
    encode_text = []
    for sentence in sentences:
        encode_text.append(GLOBAL_TOKENIZER.encode(sentence))

    return encode_text

DATA_TEXT = Preprocessor.load_text([DATASET_PATH]) # Contain all the texts of specific files
SENTENCES = Preprocessor.split_into_sentences(DATA_TEXT) # A list that contains all the sentences that meet '.!?:'

GLOBAL_TOKENIZER = Tokenizer() 
GLOBAL_TOKENIZER.build_vocabs(SENTENCES) #Build vocab from this sentences

ENCODED_SENTENCES = encode_sentences(SENTENCES) #Sentences that contains the ids of each word. Shape (Numbers of sentences, numbers of characters in each sentence)