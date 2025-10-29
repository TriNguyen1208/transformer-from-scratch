import torch
import torch.nn as nn
from torch.nn import functional as F
from app.config.constant import EMBED_DIM, NUM_LAYER, BATCH_SIZE, DEVICE, CHECKPOINT_PATH, MAX_SEQ_LEN
from app.config.config import GLOBAL_TOKENIZER
from app.layers.position_encoding import PositionalEncoding
from app.layers.block import TransformerBlock
from app.utils.utils import get_batch
from app.training.checkpoint import save_checkpoint

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = GLOBAL_TOKENIZER.get_vocab_size()
        self.token_embedding = nn.Embedding(self.vocab_size, EMBED_DIM) # shape(vocab_size, embed_dim): Map each tokens ID to an embedding vectors
        self.position_encoding = PositionalEncoding() # shape(1, max_seq_len, embed_dim): Provides positional for each position in the sequence
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYER)]) 
        self.layer_norm = nn.LayerNorm(EMBED_DIM)
        self.fc_out = nn.Linear(EMBED_DIM, self.vocab_size)

    def forward(self, input, target=None):
        token_embed = self.token_embedding(input) #shape(B, S, E)
        x = token_embed + self.position_encoding() #shape(B, S, E)
        
        pad_mask = (input == GLOBAL_TOKENIZER.PAD_ID).unsqueeze(2) #shape (B, S, 1): If it is <PAD>, it equals to True
        pad_mask = pad_mask & pad_mask.transpose(1, 2) #(shape(B, S, 1) and shape(B, 1, S)) = (B, S, S): It is True if it is <PAD_ID> 
        
        for block in self.blocks:
            x = block(x, pad_mask)

        x = self.layer_norm(x)                          # [B, S, E]
        logits = self.fc_out(x)                         # [B, S, V]

        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=GLOBAL_TOKENIZER.PAD_ID)
            return logits, loss

        return logits, None
    
    def fit(
        self,
        encoded_text, 
        optimizer, 
        num_epochs, 
        start_epoch=1
    ):
        if start_epoch == num_epochs + 1:
            print('Model is trained with full epoches')
            return

        self.train()

        steps_per_epoch = len(encoded_text) // BATCH_SIZE
        log_interval = 10

        print(f"Starting training on {DEVICE} for {num_epochs + 1 - start_epoch} epoch(s) left...")

        for epoch in range(start_epoch, num_epochs + 1):
            total_loss = 0.0
            start_index = 0

            for step in range(steps_per_epoch):
                # 1. GET BATCH (x: input, y: target/label)
                # Pass the encode_text as the data source
                x, y, current_index = get_batch(encoded_text, start_index)

                # 2. FORWARD PASS AND LOSS CALCULATION
                # The model returns logits and the calculated loss
                logits, loss = self(x, y)
                
                # 3. BACKWARD PASS AND OPTIMIZATION
                optimizer.zero_grad() # Clear previous gradients
                loss.backward()        # Compute gradients
                optimizer.step()       # Update weights
                
                total_loss += loss.item()

                if step % log_interval == 0 and step > 0:
                    print(f"  | Epoch {epoch}/{num_epochs} | Step {step}/{steps_per_epoch} | Current Batch Loss: {loss.item():.4f}")

                start_index = current_index

            # Print training status
            avg_loss = total_loss / steps_per_epoch
            print(f"\n--- Epoch {epoch} COMPLETE --- | Average Loss: {avg_loss:.4f}\n")

            # Save checkpoint
            save_checkpoint(self, optimizer, epoch, avg_loss, CHECKPOINT_PATH)

    def load(
        self, 
        checkpoint_path
    ):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval() 
        self.to(DEVICE)
        
        print("✅ Model loaded successfully and ready for inference.")
        return self
    
    def predict_next_token(
        self,
        input_token_ids
    ):
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
            logits, _ = self(x) 

        last_token_logits = logits[:, -1, :]
        
        # scaled_logits = last_token_logits / TEMPERATURE

        # if TOP_K > 0:
        #     v, i = torch.topk(scaled_logits, TOP_K)

        #     scaled_logits[scaled_logits < v[:, [-1]]] = -float('inf') 

        probs = F.softmax(last_token_logits, dim=-1)
        predicted_id = torch.multinomial(probs, num_samples=1).item()
        # predicted_id = torch.argmax(probs, dim=-1).item()
        return predicted_id
    
    def generate_response(
        self,
        prompt,
        max_new_token = 30
    ):
        self.load(CHECKPOINT_PATH)
        cur_text = prompt

        for _ in range(max_new_token):
            cur_ids = GLOBAL_TOKENIZER.encode(cur_text)
            
            next_id = self.predict_next_token(cur_ids)
            if next_id == GLOBAL_TOKENIZER.EOS_ID or next_id == GLOBAL_TOKENIZER.PAD_ID or next_id == -1:
                break

            cur_ids.append(next_id)
            cur_text += ' ' + GLOBAL_TOKENIZER.idx2word[next_id]

        return cur_text