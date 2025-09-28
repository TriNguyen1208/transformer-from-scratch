import torch
from torch.nn import functional as F
from app.config.constant import DEVICE, MAX_SEQ_LEN, CHECKPOINT_PATH
from app.training.checkpoint import loadModel
from app.config.config import GLOBAL_TOKENIZER

TEMPERATURE = 0.8
TOP_K = 50  

def predict_next_token(model, input_token_ids):
    """
    Predicts the next token ID using Argmax (Greedy Decoding).
    
    Parameters:
        model: The loaded and prepped Transformer model.
        input_token_ids (list): A sequence of token IDs (integers).
    
    Returns:
        int: The ID of the most probable next token.
    """
    
    input_ids = input_token_ids[-MAX_SEQ_LEN:] 
    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(x) 

    last_token_logits = logits[:, -1, :] 
    
    scaled_logits = last_token_logits / TEMPERATURE

    if TOP_K > 0:
        v, i = torch.topk(scaled_logits, TOP_K)

        scaled_logits[scaled_logits < v[:, [-1]]] = -float('inf') 

    probs = F.softmax(scaled_logits, dim=-1)
    predicted_id = torch.multinomial(probs, num_samples=1).item() 

    return predicted_id


model = loadModel(CHECKPOINT_PATH)
prompt = 'Cùng với đó mưa lớn do bão có khả năng gây ra ngập úng ở'
ids = GLOBAL_TOKENIZER.encode(prompt)

predicted_id = predict_next_token(model, ids)
print(GLOBAL_TOKENIZER.idx2word[predicted_id])