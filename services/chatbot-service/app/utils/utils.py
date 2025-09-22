import torch
from app.config.constant import seq_len, batch_size, device
from app.utils.tokenizer import Tokenizer

def get_batch(encode_text):
    '''
        This function is used to get batch from list of encoded_text

        Parameters
        ----------
        encode_text: list of ids of text
            List of ids of text

        Returns
        -------
            x: 2D list shape(batch_size, seq_len)
            y: 2D list that is target of x predicted shape(batch_size, seq_len)
        '''
    idx = torch.randint(len(encode_text) - seq_len, (batch_size,))
    x = torch.stack([encode_text[i: i + seq_len] for i in idx])
    y = torch.stack([encode_text[i + 1 : i + seq_len + 1] for i in idx])
    return x.to(device), y.to(device)