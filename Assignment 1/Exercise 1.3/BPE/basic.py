"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def building_merges(self, text, vocab_size, verbose=False):
        """
        Trains the tokenizer using the BPE algorithm.

        Args:
            text (str): The input text to train on.
            vocab_size (int): The desired vocabulary size (must be >= 256).
            verbose (bool): If True, prints the progress of the merges.

        Implement the training process using get_stats and merge to build the vocabulary.
        Update self.merges and self.vocab with the new tokens.
        """
        # Convert text to bytes and initialize ids
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # Encode text to bytes
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        # Track merges as we build the vocabulary
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for i in range(num_merges):
            # Get statistics on consecutive pairs
            stats = get_stats(ids)
            
            if not stats:
                break
            
            # Find the most frequent pair
            pair = max(stats, key=stats.get)
            
            # Assign new token id to this pair
            idx = 256 + i
            
            # Merge the pair in the ids list
            ids = merge(ids, pair, idx)
            
            # Store the merge
            merges[pair] = idx
            
            if verbose and (i + 1) % max(1, num_merges // 10) == 0:
                print(f"Merge {i + 1}/{num_merges}: pair {pair} -> {idx}, freq: {stats[pair]}")
        
        # Store the merges and rebuild vocab
        self.merges = merges
        self.vocab = self._build_vocab()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids