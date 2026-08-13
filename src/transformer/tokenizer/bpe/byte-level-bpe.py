"""Implementation of the Byte Pair Encoding (BPE) tokenizer."""
import json
import os
import regex as re

# Standard GPT-2 split pattern for pre-tokenization
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids, counts=None):
    """
    Counts occurrences of adjacent token ID pairs.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    Replaces all non-overlapping occurrences of `pair` in `ids` with `idx`.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class ByteLevelBPETokenizer:
    def __init__(self, pattern=None):
        self.pattern = GPT2_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {}  # tuple (p0, p1) -> merged_token_id
        self.vocab = {i: bytes([i]) for i in range(256)}  # int -> bytes
        self.special_tokens = {}  # str -> token_id
        self.inverse_special_tokens = {}  # token_id -> str

    def add_special_tokens(self, special_tokens):
        """
        Registers special tokens (e.g., {"<|endoftext|>": 100000}).
        """
        for token, idx in special_tokens.items():
            self.special_tokens[token] = idx
            self.inverse_special_tokens[idx] = token
            self.vocab[idx] = token.encode("utf-8")

    def train(self, text, vocab_size, verbose=False):
        """
        Trains the BPE tokenizer on raw text up to the specified vocabulary size.
        """
        assert vocab_size >= 256, "Vocabulary size must be at least 256."
        num_merges = vocab_size - 256

        # Step 1: Split text into chunks using pre-tokenization regex
        text_chunks = self.compiled_pattern.findall(text)

        # Step 2: Convert chunks into lists of raw UTF-8 byte integer IDs
        ids_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        # Step 3: Iteratively find the most frequent pair and merge
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk in ids_chunks:
                get_stats(chunk, stats)

            if not stats:
                break

            # Find the most frequent adjacent pair
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 1:
                break

            new_id = 256 + i
            ids_chunks = [merge(chunk, best_pair, new_id) for chunk in ids_chunks]

            # Save learned merge rank and vocabulary token
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_id} ({self.vocab[new_id]})")

        # Re-register special tokens if present
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

    def _encode_chunk(self, chunk_bytes):
        """
        Encodes a single pre-tokenized byte chunk applying learned merges in chronological rank order.
        """
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair with the lowest rank (learned earliest during training)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode(self, text, allowed_special="none"):
        """
        Encodes text into integer token IDs.
        `allowed_special` can be "all", "none", or a set of explicit special token strings.
        """
        # Determine active special tokens for this encode call
        if allowed_special == "all":
            active_specials = self.special_tokens
        elif allowed_special == "none":
            active_specials = {}
        elif isinstance(allowed_special, set):
            active_specials = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"Invalid allowed_special argument: {allowed_special}")

        if active_specials:
            # Escape and compile special token search pattern
            special_pattern = "(" + "|".join(re.escape(k) for k in active_specials.keys()) + ")"
            chunks = re.split(special_pattern, text)
        else:
            chunks = [text]

        ids = []
        for chunk in chunks:
            if chunk in active_specials:
                ids.append(active_specials[chunk])
            else:
                # Pre-tokenize standard text chunk using regex split rules
                sub_chunks = self.compiled_pattern.findall(chunk)
                for sub_chunk in sub_chunks:
                    chunk_bytes = sub_chunk.encode("utf-8")
                    ids.extend(self._encode_chunk(chunk_bytes))
        return ids

    def decode(self, ids):
        """
        Decodes a sequence of integer token IDs back to a text string.
        """
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token ID: {idx}")

        return b"".join(part_bytes).decode("utf-8", errors="replace")

    def save(self, file_prefix):
        """
        Saves the tokenizer metadata, special tokens, and merge rules to disk.
        """
        model_file = file_prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("byte-level-bpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for k, v in self.special_tokens.items():
                f.write(f"{k} {v}\n")
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")

    def load(self, model_file):
        """
        Loads trained merge rules and tokenizer configuration from disk.
        """
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            self.pattern = f.readline().strip()
            self.compiled_pattern = re.compile(self.pattern)

            num_special = int(f.readline().strip())
            self.special_tokens = {}
            self.inverse_special_tokens = {}
            for _ in range(num_special):
                line = f.readline().strip().split()
                k, v = line[0], int(line[1])
                self.special_tokens[k] = v
                self.inverse_special_tokens[v] = k

            self.merges = {}
            self.vocab = {i: bytes([i]) for i in range(256)}
            for k, v in self.special_tokens.items():
                self.vocab[v] = k.encode("utf-8")

            for line in f:
                if line.strip():
                    p0, p1, idx = map(int, line.strip().split())
                    self.merges[(p0, p1)] = idx
                    self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

if __name__ == "__main__":
    training_corpus = """
    Hello world! Building a Byte-Pair Encoding (BPE) tokenizer from scratch is fun.
    Unicode support check: नमस्ते दुनिया! Café, naïve, 123456789.
    Special token test: <|endoftext|>
    """

    # 1. Instantiate Tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # 2. Add Special Tokens
    tokenizer.add_special_tokens({"<|endoftext|>": 1000})

    # 3. Train on Corpus
    target_vocab_size = 300  # 256 base byte tokens + 44 merges
    print(f"Training tokenizer to target vocab size: {target_vocab_size}...")
    tokenizer.train(training_corpus, vocab_size=target_vocab_size, verbose=False)

    # 4. Test Sample Texts
    test_text = "Hello world! <|endoftext|> नमस्ते world!"
    
    # Encode
    encoded_ids = tokenizer.encode(test_text, allowed_special="all")
    print("\nEncoded Token IDs:")
    print(encoded_ids)

    # Decode
    decoded_text = tokenizer.decode(encoded_ids)
    print("\nDecoded Text:")
    print(decoded_text)

    # Round-trip Assertion Test
    assert test_text == decoded_text, "Round-trip assertion failed!"
    print("\nRound-trip test passed successfully!")

    # 5. Serialization Test
    tokenizer.save("bpe_test")
    
    loaded_tokenizer = ByteLevelBPETokenizer()
    loaded_tokenizer.load("bpe_test.model")
    
    assert loaded_tokenizer.encode(test_text, allowed_special="all") == encoded_ids
    assert loaded_tokenizer.decode(encoded_ids) == test_text
    print("Serialization save/load verification passed!")