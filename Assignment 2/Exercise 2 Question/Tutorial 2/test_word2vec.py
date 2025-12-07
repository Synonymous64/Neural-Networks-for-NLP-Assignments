#!/usr/bin/env python
"""
Test script for Word2Vec implementation (Task 2)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tutorial2_task2 import tokenize, generate_training_data, generate_mappings
from tutorial2_task2 import CBOW, SkipGram, train_word2vec, plot_loss

# Part (a): Tokenize and prepare training data
print("=" * 60)
print("TASK 2 - Word2Vec Implementation")
print("=" * 60)

print("\n(a) Tokenizing tutorial2.txt...")
tokens = tokenize("tutorial2.txt")
print(f"Total tokens: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")

# Generate training data with context window size 1
context_window_size = 1
center_words, context_words = generate_training_data(tokens, context_window_size)
print(f"\nTraining examples generated: {len(center_words)}")
print(f"First 5 examples:")
for i in range(min(5, len(center_words))):
    print(f"  Center: '{center_words[i]}' | Context: {context_words[i]}")

# Generate mappings
vocabulary, word2id, id2word = generate_mappings(tokens)
print(f"\nVocabulary size: {len(vocabulary)}")

# Prepare training data for models
cbow_inputs = torch.tensor([[word2id[w] for w in row] for row in context_words])
cbow_targets = torch.tensor([[word2id[w]] for w in center_words])

print(f"CBOW input shape: {cbow_inputs.shape}")
print(f"CBOW target shape: {cbow_targets.shape}")

# Part (b): Train CBOW
print("\n" + "=" * 60)
print("(b) Training CBOW Model (context window = 1)")
print("=" * 60)

embedding_dim = 50
num_epochs = 200
context_size = 2 * context_window_size

cbow_model = CBOW(len(vocabulary), context_size, embedding_dim)
print(f"\nCBOW Model Architecture:")
print(cbow_model)

print(f"\nTraining CBOW for {num_epochs} epochs...")
cbow_loss = train_word2vec(cbow_model, cbow_inputs, cbow_targets, num_epochs)
print(f"Initial loss: {cbow_loss[0]:.4f}")
print(f"Final loss: {cbow_loss[-1]:.4f}")

# Part (c): Train SkipGram
print("\n" + "=" * 60)
print("(c) Training SkipGram Model (context window = 1)")
print("=" * 60)

skipgram_inputs = cbow_targets
skipgram_targets = cbow_inputs

skipgram_model = SkipGram(len(vocabulary), context_size, embedding_dim)
print(f"\nSkipGram Model Architecture:")
print(skipgram_model)

print(f"\nTraining SkipGram for {num_epochs} epochs...")
skipgram_loss = train_word2vec(skipgram_model, skipgram_inputs, skipgram_targets, num_epochs)
print(f"Initial loss: {skipgram_loss[0]:.4f}")
print(f"Final loss: {skipgram_loss[-1]:.4f}")

# Compare training performance
print("\n" + "=" * 60)
print("Training Performance Comparison")
print("=" * 60)

plot_loss(
    'CBOW vs SkipGram Training Loss',
    [
        (cbow_loss, "CBOW", "r"),
        (skipgram_loss, "SkipGram", "b"),
    ]
)

# Test predictions
print("\n" + "=" * 60)
print("Test Predictions")
print("=" * 60)

# Test CBOW prediction
print("\nCBOW: Predicting center word from context...")
test_context_words = ["the", "movie"]
if all(word in word2id for word in test_context_words):
    context_indices = torch.tensor([[word2id[w] for w in test_context_words]])
    with torch.no_grad():
        log_probs = cbow_model(context_indices)
        predicted_idx = torch.argmax(log_probs, dim=1).item()
        predicted_word = id2word[predicted_idx]
        print(f"Context: {test_context_words}")
        print(f"Predicted center word: '{predicted_word}'")
else:
    print("Some context words not in vocabulary")

# Test SkipGram prediction
print("\nSkipGram: Predicting context words from center word...")
test_center_word = "movie"
if test_center_word in word2id:
    center_indices = torch.tensor([[word2id[test_center_word]]])
    with torch.no_grad():
        log_probs = skipgram_model(center_indices)
        predicted_indices = torch.argmax(log_probs.squeeze(), dim=1).numpy()
        predicted_words = [id2word[idx] for idx in predicted_indices]
        print(f"Center word: '{test_center_word}'")
        print(f"Predicted context words: {predicted_words}")
else:
    print(f"'{test_center_word}' not in vocabulary")

print("\n" + "=" * 60)
print("Word2Vec training complete!")
print("=" * 60)
