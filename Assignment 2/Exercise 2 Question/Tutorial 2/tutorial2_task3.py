import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split

from tutorial2_task1 import load_glove_model


def generate_mappings(embeddings):
    """
    Generate mappings between words and unique integer IDs based on the provided embeddings.

    This function takes a dictionary of word embeddings and creates two mapping dictionaries:
    - One mapping words to their corresponding integer IDs (word2id).
    - The other mapping integer IDs back to the corresponding words (id2word).

    A special token `<unk>` for unknown words is also added to the mappings with an ID of 0. 
    This token is assigned a zero vector of the same dimension as the embeddings in the input dictionary.

    Parameters:
    embeddings (dict): A dictionary where keys are words (str) and values are their corresponding 
                       embeddings (numpy arrays).

    Returns:
    tuple: A tuple containing three elements:
        - vocabulary (list of str): A list of all words in the embeddings dictionary.
        - word2id (dict): A dictionary mapping words (str) to their unique integer IDs (int).
        - id2word (dict): A dictionary mapping integer IDs (int) to their corresponding words (str).

    Example:
    >>> embeddings = {"the": np.array([1.0, 2.0]), "a": np.array([3.0, 4.0])}
    >>> vocabulary, word2id, id2word = generate_mappings(embeddings)
    >>> print(vocabulary) # Prints ['<unk>', 'the', 'a']
    >>> print(word2id)   # Prints {'<unk>': 0, 'the': 1, 'a': 2}
    >>> print(id2word)   # Prints {0: '<unk>', 1: 'the', 2: 'a'}
    """
    word2id = {}
    id2word = {}
    vocabulary = ["<unk>"] + list(embeddings.keys())

    word2id["<unk>"] = np.zeros(len(embeddings["the"]), dtype=np.float32)
    id2word[0] = "<unk>" 
    
    for index, token in enumerate(vocabulary):
        word2id[token] = index + 1
        id2word[index + 1] = token
    
    return vocabulary, word2id, id2word


def tokenize(raw_text):
    """
    Tokenize a given string into individual words.

    This function preprocesses the raw text by replacing contractions of the form "n't" with " not" 
    to standardize them. It then tokenizes the modified text into individual words. The tokenization 
    is case-insensitive as the text is converted to lowercase before tokenizing.

    Parameters:
    raw_text (str): The raw text string that needs to be tokenized.

    Returns:
    list: A list of word tokens extracted from the input text.

    Note:
    The function uses regular expressions for tokenization, thus it captures words formed of alphanumeric 
    characters and ignores punctuation.

    Example:
    >>> text = "I can't believe it's not butter!"
    >>> tokenize(text)
    ['i', 'ca', 'not', 'believe', 'it', 's', 'not', 'butter']
    """
    # normalize n't -> not
    if not isinstance(raw_text, str):
        return []
    text = re.sub(r"n't\b", " not", raw_text)
    # lowercase and extract words (alphanumeric + underscore)
    tokens = re.findall(r"\w+", text.lower())
    return tokens


def load_training_data(fname):
    """
    Load training data for sentiment analysis from a specified file.

    This function opens a file, reads its content line by line, and tokenizes each line. 
    It assumes that the file contains text data where each line represents a different review. 
    The function also creates a target tensor for sentiment labels, assigning 1 for positive 
    sentiment (which are assumed to be on the odd lines of the file) and 0 for negative sentiment 
    (on the even lines).

    Parameters:
    fname (str): The filename of the text file to be read.

    Returns:
    tuple: A tuple containing two elements:
        - words (list of list of str): A list where each element is a list of tokens 
          from each line of the file.
        - targets (torch.Tensor): A 1D tensor of the same length as `words`, containing 
          sentiment labels (1 for positive, 0 for negative).

    Example:
    >>> words, targets = load_training_data("reviews.txt")
    >>> print(words[0]) # Prints the tokens of the first line in the file
    >>> print(targets[0]) # Prints the sentiment label of the first line
    """
    words = []
    labels = []
    with open(fname, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            toks = tokenize(line)
            words.append(toks)
            # first line (idx 0) is positive (1), even index -> positive
            label = 1 if (idx % 2 == 0) else 0
            labels.append(label)

    targets = torch.tensor(labels, dtype=torch.float32)
    return words, targets


def encode_and_pad(words, word2id, max_size):
    """
    Encode and pad a list of sentences to a specified maximum size.

    This function takes a list of sentences, where each sentence is a list of words. 
    It encodes each word into its corresponding integer ID using the provided `word2id` mapping. 
    If a word is not found in the mapping, it defaults to 0 (usually representing an unknown word). 
    Each sentence is then padded with zeros to ensure that all encoded sentences have the same length, 
    specified by `max_size`.

    If a sentence is longer than `max_size`, it is truncated to fit.

    Parameters:
    words (list of list of str): A list of sentences, each sentence being a list of words.
    word2id (dict): A dictionary mapping words (str) to their unique integer IDs (int).
    max_size (int): The maximum size to which each sentence will be padded or truncated.

    Returns:
    list: A list of 1D tensors, each tensor representing an encoded and padded sentence.

    Example:
    >>> words = [["hello", "world"], ["this", "is", "a", "test"]]
    >>> word2id = {"hello": 1, "world": 2, "this": 3, "is": 4, "a": 5, "test": 6}
    >>> max_size = 5
    >>> encoded_padded_sentences = encode_and_pad(words, word2id, max_size)
    >>> print(encoded_padded_sentences)
    [tensor([1, 2, 0, 0, 0]), tensor([3, 4, 5, 6, 0])]
    """
    features = [
        torch.tensor([word2id.get(token, 0) for token in line]) for line in words
    ]
    
    features_padded = []
    
    for feature in features:
        if len(feature) > max_size:
            features_padded.append(
                feature[:max_size]
            )
        else:
            features_padded.append(
                F.pad(feature, (0, max_size - len(feature)))
            )

    return features_padded


def train(inputs, targets, num_epochs, embeddings):
    """
    Train a sentiment classifier using given inputs and targets.

    This function initializes a SentimentClassifier with the specified vocabulary size and 
    embedding dimension. It uses the Adam optimizer and binary cross-entropy loss for training. 
    The function iterates over the specified number of epochs, training the classifier with the 
    provided inputs and targets. The loss is calculated for each input-target pair, and the 
    parameters of the classifier are updated accordingly. The function also prints the loss at 
    each epoch and accumulates the total loss over all epochs.
    """
    # inputs: tensor (N, seq_len) of token ids
    # targets: tensor (N,) of floats (0/1)
    if isinstance(embeddings, np.ndarray):
        embedding_matrix = torch.from_numpy(embeddings).float()
    elif isinstance(embeddings, torch.Tensor):
        embedding_matrix = embeddings.float()
    else:
        raise ValueError("embeddings must be numpy array or torch tensor")

    vocab_size, embedding_dim = embedding_matrix.shape

    clf = SentimentClassifier(vocab_size, embedding_dim)
    # load pretrained embeddings
    with torch.no_grad():
        clf.embedding.weight.data.copy_(embedding_matrix)

    optimizer = optim.Adam(clf.parameters(), lr=0.01)
    loss_function = nn.BCELoss()
    losses = []

    inputs = inputs.long()
    targets = targets.float()

    for epoch in range(num_epochs):
        clf.train()
        optimizer.zero_grad()
        outputs = clf(inputs)  # (N, 1)
        outputs = outputs.view(-1)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {loss.item():.4f}")

    return clf, losses


class SentimentClassifier(nn.Module):
    """
    A sentiment classifier based on a Long Short-Term Memory (LSTM) network.
    """

    def __init__(self, input_size, embedding_dim, hidden_dim=256, LSTM_layers_size=2):
        super(SentimentClassifier, self).__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim, num_layers=LSTM_layers_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        # input: (batch, seq_len) of token ids
        embeds = self.embedding(input)  # (batch, seq_len, emb_dim)
        lstm_out, (hn, cn) = self.lstm(embeds)
        # hn: (num_layers, batch, hidden_dim) -> take last layer
        last_hidden = hn[-1]  # (batch, hidden_dim)
        logits = self.fc(last_hidden)  # (batch, 1)
        output = self.sig(logits)
        return output


if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "tutorial2.txt")

    print("Loading data from", data_path)
    words, targets = load_training_data(data_path)

    # split
    words_train, words_test, y_train, y_test = train_test_split(
        words, targets.numpy(), test_size=0.2, random_state=42, stratify=targets.numpy()
    )

    # try to load GloVe; fall back to random embeddings if unavailable
    glove_path = os.path.join(base_dir, "glove.6B.50d.txt")
    try:
        print("Loading GloVe from", glove_path)
        glove = load_glove_model(glove_path)
        emb_dim = next(iter(glove.values())).shape[0]
    except Exception:
        print("GloVe not found, creating random embeddings (50d)")
        glove = {}
        emb_dim = 50

    # build vocabulary from training data (include unk)
    uniq = set()
    for s in words_train:
        uniq.update(s)

    word2id = {"<unk>": 0}
    id2word = {0: "<unk>"}
    for i, w in enumerate(sorted(uniq), start=1):
        word2id[w] = i
        id2word[i] = w

    vocab_size = len(word2id)

    # build embedding matrix
    embedding_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    for w, idx in word2id.items():
        if w == "<unk>":
            continue
        if w in glove:
            embedding_matrix[idx] = glove[w]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(emb_dim,))

    # encode and pad
    max_len = max(len(s) for s in words)
    X_train_list = encode_and_pad(words_train, word2id, max_len)
    X_test_list = encode_and_pad(words_test, word2id, max_len)

    X_train = torch.stack(X_train_list)
    X_test = torch.stack(X_test_list)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Vocab: {vocab_size}, Max len: {max_len}")

    # train
    model, losses = train(X_train, y_train, num_epochs=20, embeddings=embedding_matrix)

    # evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test).view(-1)
        preds_label = (preds >= 0.5).long()
        acc = (preds_label == y_test.long()).float().mean().item()
    print(f"Test accuracy: {acc*100:.2f}%")
