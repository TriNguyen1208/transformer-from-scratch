import torch
from torch.nn import functional as F
from app.config.constant import DEVICE, MAX_SEQ_LEN, CHECKPOINT_PATH
from app.training.checkpoint import loadModel
from app.layers.transformer import TransformerModel, LibTransformerModel
from app.config.config import GLOBAL_TOKENIZER, SENTENCES

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
    
    # scaled_logits = last_token_logits / TEMPERATURE

    # if TOP_K > 0:
    #     v, i = torch.topk(scaled_logits, TOP_K)

    #     scaled_logits[scaled_logits < v[:, [-1]]] = -float('inf') 

    probs = F.softmax(last_token_logits, dim=-1)
    predicted_id = torch.multinomial(probs, num_samples=1).item() 

    return predicted_id

def generateResponse(model, start_text, max_new_tokens):
    cur_text = start_text

    for _ in range(max_new_tokens):
        cur_ids = GLOBAL_TOKENIZER.encode(cur_text)
        
        next_id = predict_next_token(model, cur_ids)

        if next_id == GLOBAL_TOKENIZER.EOS_ID or next_id == GLOBAL_TOKENIZER.PAD_ID or next_id == -1:
            break

        cur_ids.append(next_id)
        cur_text += ' ' + GLOBAL_TOKENIZER.idx2word[next_id]

    return cur_text


model = TransformerModel()
model = loadModel(model, CHECKPOINT_PATH)
prompt = 'Cùng với đó mưa lớn do bão có khả năng gây ra ngập úng'

response = generateResponse(model, prompt, 30)
print(response)

# ==================== DEBUG ====================

# for id, word in GLOBAL_TOKENIZER.idx2word.items():
#     print(f'{id}: {word}')

# for sen in SENTENCES:
#     print(GLOBAL_TOKENIZER.modifyTokens(GLOBAL_TOKENIZER.tokenizeWords(sen)))
#     print('=' * 20)