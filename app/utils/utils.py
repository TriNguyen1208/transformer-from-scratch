import torch
from app.config.constant import BATCH_SIZE, DEVICE
from app.config.config import GLOBAL_TOKENIZER

def get_batch(encode_text, start_index):
    '''
        This function is used to get batch from list of encoded_text

        Parameters
        ----------
        encode_text: list of ids of text
            List of ids of text

        Returns
        -------
            x: 2D list shape(BATCH_SIZE, MAX_SEQ_LEN)
            y: 2D list that is target of x predicted shape(BATCH_SIZE, MAX_SEQ_LEN)
        '''
    end_index = min(start_index + BATCH_SIZE, len(encode_text))

    x = torch.stack([torch.tensor(encode_text[int(i)], dtype=torch.long) for i in range(start_index, end_index)])
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = GLOBAL_TOKENIZER.word2idx['<PAD>']

    return x.to(DEVICE), y.to(DEVICE), end_index
