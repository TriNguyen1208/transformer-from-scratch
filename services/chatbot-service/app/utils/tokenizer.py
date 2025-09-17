import re

class Tokenizer:
    def __init__(self):
        self.special_tokens = ['<UNK>', '<EOS>']
        self.word_to_idx = {}
        self.idx_to_word = {}

        self.UNK_ID = None
        self.EOS_ID = None

    def tokenizeWords(self, text):
        '''
        This function is used to split word from the whole text

        Parameters
        ----------
        text: string
            String that contains txt data

        Returns
        -------
            The list that contains each token including words and punctuations
        '''

        text = text.lower()
        tokens = re.findall(r"\w+|[.,!?;]", text, flags=re.UNICODE)
        return tokens

    def buildVocabs(self, text):
        '''
        This function is used to get the vocabs from the texts

        Parameters
        ----------
        text: string
            String that contains txt data
        '''

        # Split the words
        words = self.tokenizeWords(text)

        # Unique vocab set + special tokens
        vocabs = self.special_tokens + sorted(set(words))

        # Build dicts
        self.word2idx = {w: i for i, w in enumerate(vocabs)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        # Save IDs for quick access
        self.UNK_ID = self.word2idx["<UNK>"]
        self.EOS_ID = self.word2idx["<EOS>"]

    def encode(self, text, add_eos = True):
        '''
        This function is used to convert words into IDs

        Parameters
        ----------
        add_eos: boolean (True as default)
            True if you want to add <EOS> token at the end of sentence
        text: string
            The text to encode

        Returns
        -------
        List that contains the IDs of corresponding words
        '''

        tokens = self.tokenizeWords(text)
        ids = [self.word2idx.get(token, self.UNK_ID) for token in tokens]

        if add_eos == True:
            ids.append(self.EOS_ID)

        return ids
    
    def decode(self, ids, remove_eos = True):
        '''
        This functionis used to convert IDs into words

        Parameters
        ----------
        ids: list of ids converted from the encoding process

        Returns
        -------
        The text after decoding
        '''

        words = []
        for idx in ids:
            if remove_eos == True and idx == self.EOS_ID:
                break

            words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)