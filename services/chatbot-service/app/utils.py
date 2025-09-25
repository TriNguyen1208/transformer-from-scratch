import torch
from app.config.constant import BATCH_SIZE, DEVICE
from app.config.config import GLOBAL_TOKENIZER

def getBatch(encode_text):
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
    idx = torch.randint(0, len(encode_text), (BATCH_SIZE,))

    # x = torch.stack([encode_text[i: i + MAX_SEQ_LEN] for i in idx])
    # y = torch.stack([encode_text[i + 1 : i + MAX_SEQ_LEN + 1] for i in idx])

    x = torch.stack([torch.tensor(encode_text[int(i)], dtype=torch.long) for i in idx])
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = GLOBAL_TOKENIZER.word2idx['<PAD>']

    return x.to(DEVICE), y.to(DEVICE)
