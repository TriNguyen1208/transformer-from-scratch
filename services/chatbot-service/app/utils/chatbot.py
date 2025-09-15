import re
import torch
import torch.nn as nn

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)   # collapse spaces
    return text

def word_tokenize(text):
    # lowercase optional
    text = text.lower()
    # remove punctuation (optional)
    text = re.sub(r"[^a-zA-Z0-9À-ỹ\s]", "", text)
    return text.split()

def encode(sentence, dict):
    return [dict[word] for word in sentence.split(' ')]

data_raw = ''
with open('dataset.txt', 'r', encoding='utf-8') as f:
    data_raw = f.read()

data = clean_text(data_raw)
words = word_tokenize(data)

vocabs = set(words)
encode_dict = {text: i for i, text in enumerate(vocabs)}
decode_dict = {i: text for i, text in enumerate(vocabs)}


vocab_size = len(vocabs)
embedding_dim = 64

embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Example: encode a sentence to IDs
sentence_ids = torch.tensor(encode("Xin chào bạn".lower(), encode_dict))  # [2, 3, 4]

# Get embeddings
embedded = embedding(sentence_ids)  # shape: [seq_len, embedding_dim]

print(sentence_ids)  # e.g., torch.Size([3, 128])