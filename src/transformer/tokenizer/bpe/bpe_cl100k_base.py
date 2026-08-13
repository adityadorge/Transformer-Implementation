""" 
To upgrade your BPE tokenizer from GPT-2 style to GPT-4 (cl100k_base) or GPT-4o (o200k_base) standards, four major architecture and algorithmic improvements are required:
1. Updated GPT-4 Regex Pre-tokenization Pattern:
    - Case-Insensitive Contractions: Captures 'S, 's, 'RE, 're uniformly ((?i:...)).
    - Digit Chunking (\p{N}{1,3}): Restricts number merges to a maximum of 3 digits at a time. This stops long numerical sequences (e.g., 123456789) from merging into arbitrary single tokens, significantly boosting code and math capabilities.
    - Whitespace & Newline Isolation: Keeps trailing newlines and indentations separate to maintain syntax awareness for Python/YAML.
2. Expanded Vocabulary Scale: GPT-4 increases vocabulary size to ~100k tokens (cl100k_base), while GPT-4o expands to ~200k tokens (o200k_base). A larger vocabulary improves compression ratio by up to 2x for non-English languages and code.
3. Fill-In-the-Middle (FIM) & System Control Tokens: Full support for code completion control tokens (<|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>, <|endofprompt|>) alongside standard sequence markers.
4. Rank-Based Encoding Optimizations: Accelerated chunk encoding using $O(1)$ priority lookups on trained merge ranks rather than re-computing pair frequency histograms.
"""

import json
import os
import regex as re

# Official GPT-4 Pre-tokenization Regex Pattern (tiktoken cl100k_base)
GPT4_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# Standard GPT-4 Special / Control Tokens
GPT4_SPECIAL_TOKENS = {
    "<|endoftext|>": 100000,
    "<|fim_prefix|>": 100001,
    "<|fim_middle|>": 100002,
    "<|fim_suffix|>": 100003,
    "<|endofprompt|>": 100004,
}


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


class GPT4BPETokenizer:
    def __init__(self, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {}  # tuple (p0, p1) -> token_id (lower token_id = earlier merge rank)
        self.vocab = {i: bytes([i]) for i in range(256)}  # int -> bytes
        self.special_tokens = {}
        self.inverse_special_tokens = {}

        # Register default GPT-4 special tokens
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def register_special_tokens(self, special_tokens_dict):
        """
        Registers control and special tokens with custom fixed token IDs.
        """
        for token, idx in special_tokens_dict.items():
            self.special_tokens[token] = idx
            self.inverse_special_tokens[idx] = token
            self.vocab[idx] = token.encode("utf-8")

    def train(self, text, vocab_size=100000, verbose=False):
        """
        Trains the BPE tokenizer on raw text up to the specified target vocabulary size.
        """
        assert vocab_size >= 256, "Vocabulary size must be at least 256."
        num_merges = vocab_size - 256

        # Step 1: Pre-tokenize text into structural chunks using GPT-4 regex rules
        text_chunks = self.compiled_pattern.findall(text)

        # Step 2: Convert chunks into raw UTF-8 byte integer lists
        ids_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        # Step 3: Train BPE merges iteratively
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk in ids_chunks:
                get_stats(chunk, stats)

            if not stats:
                break

            # Select pair with highest global frequency
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 1:
                break

            new_id = 256 + i
            ids_chunks = [merge(chunk, best_pair, new_id) for chunk in ids_chunks]

            # Save learned rank and vocabulary byte sequence
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i + 1) % 1000 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_id}")

        # Re-apply special tokens to vocabulary
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

    def _encode_chunk(self, chunk_bytes):
        """
        Fast rank-ordered pair merging on individual pre-tokenized chunks.
        """
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            # Generate adjacent candidate pairs
            pairs = list(zip(ids, ids[1:]))
            
            # Find the candidate pair with the lowest merge token ID (trained earliest)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break
                
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode(self, text, allowed_special="all"):
        """
        Encodes input string into token IDs with special token parsing.
        """
        if allowed_special == "all":
            active_specials = self.special_tokens
        elif allowed_special == "none":
            active_specials = {}
        elif isinstance(allowed_special, set):
            active_specials = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"Invalid allowed_special configuration: {allowed_special}")

        if active_specials:
            # Build regex pattern to catch special tokens safely before BPE
            special_pattern = "(" + "|".join(re.escape(k) for k in active_specials.keys()) + ")"
            chunks = re.split(special_pattern, text)
        else:
            chunks = [text]

        ids = []
        for chunk in chunks:
            if chunk in active_specials:
                ids.append(active_specials[chunk])
            else:
                # Pre-tokenize sub-chunk and apply BPE merges
                sub_chunks = self.compiled_pattern.findall(chunk)
                for sub_chunk in sub_chunks:
                    chunk_bytes = sub_chunk.encode("utf-8")
                    ids.extend(self._encode_chunk(chunk_bytes))
        return ids

    def decode(self, ids):
        """
        Decodes token IDs back to a raw text string.
        """
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid Token ID: {idx}")

        return b"".join(part_bytes).decode("utf-8", errors="replace")

    def save(self, file_prefix):
        """
        Saves tokenizer model parameters and merge rules to disk.
        """
        model_file = file_prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("gpt4-bpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for k, v in self.special_tokens.items():
                f.write(f"{k} {v}\n")
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")

    def load(self, model_file):
        """
        Loads pre-trained tokenizer state from a saved model file.
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


# -----------------------------------------------------------------------------
# GPT-4 Tokenizer Demonstration & Verification
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    code_corpus = """
    def compute_sum(a: int, b: int) -> int:
        # GPT-4 handles numbers (e.g. 123456789) by splitting every 1-3 digits!
        val_1 = 123
        val_2 = 456789
        return a + b + val_1 + val_2

    <|fim_prefix|>def add(a, b):<|fim_suffix|> return a + b<|fim_middle|>
    """

    # 1. Instantiate and Train
    tokenizer = GPT4BPETokenizer()
    tokenizer.train(code_corpus, vocab_size=320, verbose=False)

    # 2. Test Number Chunking & FIM Special Tokens
    sample_text = "<|fim_prefix|>val = 123456789<|fim_suffix|>"
    encoded_ids = tokenizer.encode(sample_text, allowed_special="all")
    decoded_text = tokenizer.decode(encoded_ids)

    print("Encoded Token IDs:")
    print(encoded_ids)
    print("\nDecoded Text:")
    print(decoded_text)

    # 3. Assert exact round-trip reconstruction
    assert sample_text == decoded_text, "Round-trip assertion failed!"
    print("\nRound-trip test passed successfully!")