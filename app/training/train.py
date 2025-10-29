# import os
# from app.config.constant import DEVICE, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, CHECKPOINT_PATH
# from app.layers.transformer import TransformerModel
# from app.utils.utils import get_batch
# from app.config.config import ENCODED_SENTENCES
# from app.training.checkpoint import save_checkpoint, load_checkpoint_epoch
# from sklearn.metrics import accuracy_score
# from torch.nn import functional as F
# import torch

# def train_model(model, encoded_text, optimizer, num_epochs, start_epoch=1):
#     if start_epoch == num_epochs + 1:
#         print('Model is trained with full epoches')
#         return

#     model.train()

#     steps_per_epoch = len(encoded_text) // BATCH_SIZE
#     log_interval = 10

#     print(f"Starting training on {DEVICE} for {num_epochs + 1 - start_epoch} epoch(s) left...")

#     for epoch in range(start_epoch, num_epochs + 1):
#         total_loss = 0.0
#         start_index = 0

#         for step in range(steps_per_epoch):
#             # 1. GET BATCH (x: input, y: target/label)
#             # Pass the encode_text as the data source
#             x, y, current_index = get_batch(encoded_text, start_index)

#             # 2. FORWARD PASS AND LOSS CALCULATION
#             # The model returns logits and the calculated loss
#             logits, loss = model(x, y)
            
#             # 3. BACKWARD PASS AND OPTIMIZATION
#             optimizer.zero_grad() # Clear previous gradients
#             loss.backward()        # Compute gradients
#             optimizer.step()       # Update weights
            
#             total_loss += loss.item()

#             if step % log_interval == 0 and step > 0:
#                 print(f"  | Epoch {epoch}/{num_epochs} | Step {step}/{steps_per_epoch} | Current Batch Loss: {loss.item():.4f}")

#             start_index = current_index

#         # Print training status
#         avg_loss = total_loss / steps_per_epoch
#         print(f"\n--- Epoch {epoch} COMPLETE --- | Average Loss: {avg_loss:.4f}\n")

#         # Save checkpoint
#         save_checkpoint(model, optimizer, epoch, avg_loss, CHECKPOINT_PATH)


# model = TransformerModel()
# model.to(DEVICE)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# start_epoch = 1

# try:
#     if os.path.exists(CHECKPOINT_PATH):
#         start_epoch = load_checkpoint_epoch(model, optimizer, CHECKPOINT_PATH)

# except Exception as e:
#     print(f'Checkpoint file does not exist! Starting from epoch 1. Exception: {e}')

# train_model(model, ENCODED_SENTENCES, optimizer, NUM_EPOCHS, start_epoch)

from app.models.transformer import TransformerModel
from app.config.config import ENCODED_SENTENCES
from app.config.constant import NUM_EPOCHS, DEVICE, LEARNING_RATE, CHECKPOINT_PATH
import torch
import os
from app.training.checkpoint import load_checkpoint_epoch

model = TransformerModel()
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

start_epoch = 1

try:
    if os.path.exists(CHECKPOINT_PATH):
        start_epoch = load_checkpoint_epoch(model, optimizer, CHECKPOINT_PATH)

except Exception as e:
    print(f'Checkpoint file does not exist! Starting from epoch 1. Exception: {e}')
model.fit(ENCODED_SENTENCES, optimizer, NUM_EPOCHS, start_epoch)