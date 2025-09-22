from app.utils.processor import Preprocessor
from app.utils.tokenizer import Tokenizer
import os
from app.config.constant import BASE_DIR, FILE_NAME, generate_every, max_gen_len, device, max_iter, learning_rate, text_encode, checkpoint_path, tokenizer
from app.models.transformer import TransformerModel
from app.utils.utils import get_batch
import torch


# model = TransformerModel()
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# for _ in range(max_iter):
#     x_batch, y_batch = get_batch(text_encode)
#     logits, loss = model(x_batch, y_batch)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     print("Loss:", loss.item())
#     print("=" * 50)

# Assume TransformerModel and get_batch are defined
model = TransformerModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(1, max_iter + 1):
    # ===== Training =====
    model.train()
    x_batch, y_batch = get_batch(text_encode)
    x_batch = x_batch[:, :50]  # truncate to 50 tokens
    y_batch = y_batch[:, :50]  # same for targets

    logits, loss = model(x_batch, y_batch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # print(f"Step {step} - Loss: {loss.item()}")
    # print("=" * 50)
    
    # ===== Text Generation =====
    if step % generate_every == 0:
        model.eval()
        prompt = ("Hạnh thuộc loại người liến thoắng, nhanh nhẩu, miệng nói tay làm. Nga chưa kịp có ý kiến, nó đã vớ lấy chiếc cặp của cô bạn mới nhét vào ngăn bàn. Trước sự nhiệt tình của Hạnh, Nga chỉ biết lúng túng đứng nhìn. Hạnh đang mỉm cười nhìn")
        
        input_ids = torch.tensor([tokenizer.encode(prompt, False)], device=device)
        generated = input_ids
        # print(len(generated[0]))


        with torch.no_grad():
            logits, _ = model(generated, None)  # no y_batch during generation
            next_token_logits = logits[:, -1, :]
        
            # Greedy decoding
            # prob = torch.softmax(next_token_logits / 1.0, dim=-1)
            # next_token = torch.multinomial(prob, num_samples=1)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        # for _ in range(max_gen_len):
        #     with torch.no_grad():
        #         logits, _ = model(generated, None)  # no y_batch during generation
        #         next_token_logits = logits[:, -1, :]
                
        #         # Greedy decoding
        #         next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
        #         generated = torch.cat([generated, next_token], dim=1)

        generated_text = tokenizer.decode(generated[0].tolist(), False)
        print(f"Generated text after {step} steps:\n{generated_text}")
        print("=" * 50)

# ===== Save checkpoint =====
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
# }, checkpoint_path)

# print(f"Model checkpoint saved to {checkpoint_path}")

