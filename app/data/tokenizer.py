import re
from app.config.constant import MAX_SEQ_LEN

class Tokenizer:
    def __init__(self):
        self.special_tokens = ['<UNK>', '<EOS>', '<PAD>']
        self.word_to_idx = {}
        self.idx_to_word = {}

        self.UNK_ID = None
        self.EOS_ID = None
        self.PAD_ID = None


    def tokenize_words(self, sentence):
        '''
        This function is used to split word from a sentence

        Parameters
        ----------
        sentence: string
            A sentence contains multiples of words, even punctuations

        Returns
        -------
            The list that contains each token including words
        '''
       
        return re.split(r"(?<!\d),(?!\d)|[^\wÀ-ỹ\-]+", sentence.lower())

    def modify_tokens(self, tokens):
        '''
        This function is used to pad and truncate the list of tokens of each sentence to match MAX_SEQ_LEN

        Parameters
        ----------
        tokens: list
            A list contains tokens of a sentence

        Returns
        -------
            The list that contains the initial tokens and special tokens <PAD>, <EOS>
        '''
        
        tokens = [token for token in tokens if (token != '' and token != '-')]

        if len(tokens) >= MAX_SEQ_LEN:
           tokens = tokens[:MAX_SEQ_LEN]
           tokens[MAX_SEQ_LEN - 1] = '<EOS>'
        elif len(tokens) < MAX_SEQ_LEN:
            tokens = tokens + ['<EOS>'] + ['<PAD>'] * (MAX_SEQ_LEN - len(tokens) - 1)

        return tokens


    def build_vocabs(self, sentences):
        '''
        This function is used to get the vocabs from the list of sub-lists containing tokens of a sentence

        Parameters
        ----------
        sentences: list
            list of sentences
        '''

        words = []

        # Get word in data
        for sentence in sentences:
            tokens = self.modify_tokens(self.tokenize_words(sentence))
            words.extend(tokens)

        # Unique vocab set + special tokens
        vocabs = sorted(set(self.special_tokens + words))

        # Build dicts
        self.word2idx = {w: i for i, w in enumerate(vocabs)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        # Save IDs for quick access
        self.UNK_ID = self.word2idx['<UNK>']
        self.EOS_ID = self.word2idx['<EOS>']
        self.PAD_ID = self.word2idx['<PAD>']


    def get_vocab_size(self):
        return len(self.idx2word)


    def encode(self, text):
        '''
        This function is used to convert words into IDs

        Parameters
        ----------
        text: string
            The text to encode

        Returns
        -------
        List that contains the IDs of corresponding words
        '''

        tokens = self.modify_tokens(self.tokenize_words(text))
        ids = [self.word2idx.get(token, self.UNK_ID) for token in tokens]

        return ids
    
    def decode(self, ids):
        '''
        This function is used to convert IDs into words

        Parameters
        ----------
        ids: list of ids converted from the encoding process

        Returns
        -------
        The text after decoding
        '''

        words = []
        for idx in ids:
            words.append(self.idx2word.get(idx, '<UNK>'))

        return words