# Task 1: Linear- and Nonlinear Models for Language Classification

## Problem Overview
Given two lists of 1,000 Spanish and 1,000 English words, we need to build a classifier that predicts whether a new word is English or Spanish using vectorization and machine learning models.

---

## (a) Vectorization Mapping for Words

### Description of Vectorization Approaches

Several sensible mapping functions $\phi$ can be used to convert words into vectors:

#### **1. Character N-gram Representation** (Recommended)
Map each word to a vector where each dimension represents the frequency of a character n-gram (e.g., bigrams, trigrams).

**Example for word "hola" (Spanish):**
- Extract trigrams: "hol", "ola"
- Create a vector where each position corresponds to a specific trigram
- Value = frequency of that trigram in the word

**Advantages:**
- Captures language-specific character patterns
- Spanish and English have different common n-gram distributions
- Simple and effective for language identification

#### **2. One-Hot Encoding of Character Sequences**
Represent each character's position and ASCII value as features.

$$\phi(\text{word}) = [\text{char}_1, \text{char}_2, \ldots, \text{char}_n, \text{length}]$$

#### **3. Linguistic Features**
- Vowel ratio (a, e, i, o, u vs. y, w in English)
- Common prefix/suffix patterns
- Letter frequency analysis
- Presence of diacritics (ñ, á, é, í, ó, ú, ü common in Spanish)

#### **4. TF-IDF or Word Embeddings**
- Use pre-trained embeddings (Word2Vec, FastText)
- Language-agnostic but still captures semantic patterns

### Recommended Approach: Character N-gram Vectors

For this problem, **character trigram frequencies** work well because:
1. Spanish and English have distinctly different n-gram distributions
2. Example: "th" is common in English but rare in Spanish
3. Spanish has specific patterns like "ción", "mente" which are rare in English
4. This method is language-independent and doesn't require diacritics

---

## (b) Linear vs. Non-linear Models: Which to Use?

### Answer: **Non-linear approach (Neural Networks) is more appropriate**

### Justification

#### Why Linear Models May Not Be Sufficient:

1. **Feature Space Complexity**
   - N-gram feature spaces are high-dimensional (potentially 1000s of dimensions)
   - Language boundaries are not necessarily linearly separable in raw feature space
   - Spanish and English words overlap in character usage (both use common letters)

2. **Non-linear Interactions**
   - Language is determined by complex interactions between features, not just linear combinations
   - Example: The presence of "q" followed by "u" has different meanings in Spanish vs. English
   - Simple weighted sums of n-gram frequencies may not capture these interactions

3. **Language Structure**
   - Languages have hierarchical patterns (phonetics → syllables → morphemes → words)
   - Linear models cannot capture this hierarchical structure

#### Why Non-linear Models (Neural Networks) Are Better:

1. **Universal Approximation**
   - Neural networks can learn arbitrary non-linear decision boundaries
   - Can discover complex patterns combining multiple features

2. **Hidden Representations**
   - Multiple hidden layers can learn intermediate linguistic features
   - First layer might learn character patterns
   - Deeper layers combine these into word-level patterns

3. **Practical Performance**
   - Modern deep learning has proven superior for NLP tasks
   - Can handle high-dimensional feature spaces effectively

### Comparison Table

| Aspect | Linear Model (SVM) | Neural Network |
|--------|-------------------|-----------------|
| Decision Boundary | Linear hyperplane | Non-linear, complex surfaces |
| Feature Interactions | Limited (linear) | Unlimited (non-linear) |
| Training Time | Fast | Moderate to slow |
| Interpretability | High | Low |
| Accuracy for Language Classification | Moderate | High |
| Flexibility | Rigid | Highly flexible |

---

## (c) Neural Network Classifier

### Network Architecture

#### **Input Layer**
- **Dimension**: $n$ (number of features)
- **Representation**: Character n-gram frequency vectors
- Example: If vocabulary has 500 unique trigrams, input dimension = 500
- **Values**: Normalized n-gram frequencies (0 to 1)

#### **Hidden Layers**
```
Input Layer (n neurons)
    ↓
Hidden Layer 1 (e.g., 128 neurons, ReLU activation)
    ↓
Hidden Layer 2 (e.g., 64 neurons, ReLU activation)
    ↓
Hidden Layer 3 (e.g., 32 neurons, ReLU activation)
    ↓
Output Layer (1 neuron, Sigmoid activation)
```

**Typical Architecture:**
- 2-4 hidden layers with decreasing neuron counts
- ReLU activation for hidden layers: $f(x) = \max(0, x)$
- Sigmoid activation for output: $\sigma(x) = \frac{1}{1 + e^{-x}}$

#### **Output Layer**
- **Dimension**: 1 neuron
- **Activation**: Sigmoid function
- **Interpretation**: 
  - Output close to 1 → English
  - Output close to 0 → Spanish
  
Alternatively, use 2 neurons with softmax for multi-class classification.

### Training Procedure

#### **Step 1: Data Preparation**
```
1. Split data into training, validation, and test sets
   - Training: 1,600 words (800 English + 800 Spanish) - 80%
   - Validation: 200 words (100 English + 100 Spanish) - 10%
   - Test: 200 words (100 English + 100 Spanish) - 10%

2. Vectorize all words using n-gram mapping
   ϕ(word) = [count_trigram_1, count_trigram_2, ..., count_trigram_n]

3. Normalize feature vectors
   - Scale to [0, 1] or standardize (mean=0, std=1)

4. Create binary labels
   - English: y = 1
   - Spanish: y = 0
```

#### **Step 2: Initialize Network**
```
1. Random weight initialization (He initialization for ReLU)
2. Set bias terms to small values
3. Choose hyperparameters:
   - Learning rate: 0.001 to 0.01
   - Batch size: 32 or 64
   - Number of epochs: 100-500
   - Optimizer: Adam, SGD, or RMSprop
```

#### **Step 3: Training Loop**
```
For each epoch:
    For each batch of training data:
        1. Forward pass: Calculate network output
           ŷ = σ(W_out · ReLU(W_2 · ReLU(W_1 · x + b_1) + b_2) + b_out)
        
        2. Calculate loss (Binary Cross-Entropy)
           L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
        
        3. Backward pass: Compute gradients using backpropagation
           ∂L/∂W, ∂L/∂b for all layers
        
        4. Update weights using gradient descent
           W ← W - α · ∂L/∂W
           b ← b - α · ∂L/∂b
    
    5. Evaluate on validation set
       - Calculate validation accuracy
       - Monitor for overfitting
    
    6. Early stopping (if validation loss doesn't improve)
```

#### **Step 4: Loss Function and Optimization**

**Binary Cross-Entropy Loss:**
$$L(\mathbf{W}) = -\frac{1}{m}\sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

Where:
- $m$ = number of training samples
- $y_i$ = true label (0 or 1)
- $\hat{y}_i$ = predicted probability
- $\mathbf{W}$ = all network parameters

**Optimization Algorithm:** Adam (Adaptive Moment Estimation)
- Adapts learning rate for each parameter
- Combines momentum and RMSprop advantages
- Generally converges faster than vanilla SGD

#### **Step 5: Evaluation**

```
After training:
1. Evaluate on test set
   - Calculate accuracy, precision, recall, F1-score
   - Confusion matrix
   
2. Metrics:
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   Precision = TP / (TP + FP)
   Recall = TP / (TP + FN)
   F1 = 2 · (Precision · Recall) / (Precision + Recall)

3. Generate classification report
```

### Pseudocode Summary

```python
# Initialize
network = NeuralNetwork(layers=[n_features, 128, 64, 32, 1])
optimizer = Adam(learning_rate=0.001)
loss_fn = BinaryCrossEntropy()

# Training
for epoch in range(num_epochs):
    for batch in training_loader:
        # Forward pass
        predictions = network.forward(batch.X)
        loss = loss_fn(predictions, batch.y)
        
        # Backward pass
        gradients = loss.backward()
        
        # Update weights
        optimizer.step(network.parameters(), gradients)
    
    # Validation
    val_loss = evaluate(network, validation_set)
    if val_loss improved:
        save_checkpoint()
    else if no improvement for N epochs:
        break  # Early stopping

# Test
test_accuracy = evaluate(network, test_set)
```

### Expected Performance

- **Training Accuracy**: 95-99% (neural network can memorize)
- **Validation Accuracy**: 90-95%
- **Test Accuracy**: 85-92%
- Differences indicate some overfitting, which is normal

---

## Summary

| Component | Answer |
|-----------|--------|
| **Vectorization** | Character n-gram (trigram) frequency vectors |
| **Model Type** | Neural Network (non-linear) |
| **Why NN?** | Language patterns are non-linear; NN learns hierarchical representations |
| **Input** | Vectorized word features (dimension: ~500-2000) |
| **Output** | Single neuron with sigmoid (probability of being English) |
| **Training** | Backpropagation with binary cross-entropy loss, Adam optimizer |
