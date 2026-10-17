#DATA CLEANING
file_path = "dataset.txt.txt"
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
text = text.lower()
punctuation = ".,!?;:\"'()[]{}/*-_+=#@$%^&\\|<>~'–—"
chars_list = []
for char in text:
    if char in punctuation:
        chars_list.append(" ")
    else:
        chars_list.append(char)
text = "".join(chars_list)
words = text.split()
cleaned_text = " ".join(words)

#TOKENISATION
tokens = cleaned_text.split()
vocab = []
for token in tokens:
    if token not in vocab:
        vocab.append(token)
word_to_id = {}
id_to_word = {}
for idx, word in enumerate(vocab):
    word_to_id[word] = idx
    id_to_word[idx] = word

token_ids = []
for token in tokens:
    token_ids.append(word_to_id[token])
print("Total number of tokens:", len(tokens))
print("Vocabulary size (unique tokens):", len(vocab))

#window generation
window_size = 2
pair = []

for i in range(len(token_ids)):
    target = token_ids[i]
    start = max(0, i - window_size)
    end = min(len(token_ids), i + window_size + 1)
    
    for j in range(start, end):
        if i != j:
            context = token_ids[j]
            pair.append([target, context])
print("Total training pairs generated:", len(pair))

#populating vectors with random value of 16 dim
import random
embedding_dim = 16
vocab_size = len(vocab)
embedding_table = []
for i in range(vocab_size):
    row = [random.uniform(-0.5, 0.5) for _ in range(embedding_dim)]
    embedding_table.append(row)

epochs = 20
lr = 0.01
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(pair)):
        target_id = pair[i][0]
        context_id = pair[i][1]
        t = embedding_table[target_id]
        h = embedding_table[context_id]
        loss = 0
        for k in range(16):
            diff = h[k] - t[k]
            loss += 0.5 * (diff) ** 2
            h[k] = h[k] - (lr * diff)
            t[k] = t[k] + (lr * diff)
        total_loss += loss
    print(f"Epoch {epoch + 1} completed. Loss: {total_loss:.4f}")

import numpy as np
embedding_matrix = np.array(embedding_table)
query_word = input("enter query")
top_n = 3
if query_word not in word_to_id:
    print(f"Word '{query_word}' not found in vocabulary.")
else:
    query_id = word_to_id[query_word]
    query_vec = embedding_matrix[query_id]
    dot_products = np.dot(embedding_matrix, query_vec)
    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(embedding_matrix, axis=1)
    similarities = dot_products / (matrix_norms * query_norm + 1e-8)
    sorted_indices = np.argsort(similarities)[::-1]
    print(f"Top {top_n} words similar to '{query_word}':")
    for idx in sorted_indices[1 : top_n + 1]:
        word = id_to_word[idx]
        score = similarities[idx]
        print(word)
