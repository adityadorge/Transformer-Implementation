"""
    Architectural Breakthroughs Beyond Standard BPE (2025–2026)
    Recent advances eliminate traditional subword token boundaries:
    - SuperBPE (Liu et al., 2025): Performs a second training pass that drops word-boundary restrictions altogether. This produces up to 33% fewer tokens per corpus and delivers a +4.0% average gain across downstream LLM benchmarks.
    - BoundlessBPE (Schmidt et al., 2025): Allows multi-word phrases (e.g., "machine learning", "of the") to merge into single tokens in a single training pass, improving byte compression by up to 20%.
    - Byte-Latent Transformers (BLT - Meta, 2025): Shifts away from fixed token vocabularies by learning dynamic, continuous latent representations directly on raw byte streams.

"""

import heapq
import json
import os
import regex as re

# Official GPT-4 / GPT-4o Regex Pre-tokenization Pattern
GPT4O_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# Default Control / System Special Tokens
DEFAULT_SPECIAL_TOKENS = {
    "<|endoftext|>": 200000,
    "<|fim_prefix|>": 200001,
    "<|fim_middle|>": 200002,
    "<|fim_suffix|>": 200003,
    "<|endofprompt|>": 200004,
}


class Node:
    """Doubly Linked List Node for O(1) Token Insertion and Splice Operations."""
    __slots__ = ("val", "prev", "next")

    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None


def get_stats(ids, counts=None):
    """Counts pair frequencies across integer sequence chunks."""
    counts = {} if counts is None else counts
    for p0, p1 in zip(ids, ids[1:]):
        pair = (p0, p1)
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_list(ids, pair, idx):
    """Naive merge fallback used during trainer sequence updates."""
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


class OptimizedBPETokenizer:
    """
    Production-Ready Byte-Level BPE Tokenizer implementing:
    - GPT-4 / GPT-4o regex pre-tokenization boundary splitting
    - Fast Min-Heap + Doubly Linked List priority merge encoding
    - Comprehensive special token parsing
    - Complete save/load model serialization
    """

    def __init__(self, pattern=None):
        self.pattern = GPT4O_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {}  # tuple (p0, p1) -> merged_token_id
        self.vocab = {i: bytes([i]) for i in range(256)}  # int -> bytes
        self.special_tokens = {}
        self.inverse_special_tokens = {}

        self.register_special_tokens(DEFAULT_SPECIAL_TOKENS)

    def register_special_tokens(self, special_tokens_dict):
        """Registers system and control special tokens."""
        for token, idx in special_tokens_dict.items():
            self.special_tokens[token] = idx
            self.inverse_special_tokens[idx] = token
            self.vocab[idx] = token.encode("utf-8")

    def train(self, text, vocab_size=200000, verbose=False):
        """Trains BPE merge rules on raw input text up to target vocab_size."""
        assert vocab_size >= 256, "Vocabulary size must be at least 256."
        num_merges = vocab_size - 256

        # Step 1: Split raw text using regex boundaries
        text_chunks = self.compiled_pattern.findall(text)

        # Step 2: Convert text chunks into raw byte token ID lists
        ids_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        # Step 3: Run iterative merge loop
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk in ids_chunks:
                get_stats(chunk, stats)

            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 1:
                break

            new_id = 256 + i
            ids_chunks = [merge_list(chunk, best_pair, new_id) for chunk in ids_chunks]

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i + 1) % 1000 == 0:
                print(f"Learned Merge {i+1}/{num_merges}: {best_pair} -> {new_id}")

        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

    def _encode_chunk_fast(self, chunk_bytes):
        """
        Fast priority-queue chunk encoding using Doubly Linked List + Min-Heap.
        Achieves optimal merge performance without scanning lists redundantly.
        """
        if len(chunk_bytes) < 2:
            return list(chunk_bytes)

        # 1. Build Doubly Linked List of raw bytes
        head = Node(chunk_bytes[0])
        curr = head
        for b in chunk_bytes[1:]:
            node = Node(b)
            curr.next = node
            node.prev = curr
            curr = node

        # 2. Populate Min-Heap with candidate merge pairs ordered by merge rank
        heap = []
        curr = head
        while curr and curr.next:
            pair = (curr.val, curr.next.val)
            if pair in self.merges:
                rank = self.merges[pair]
                # Heap elements: (rank, node_id, left_node)
                heapq.heappush(heap, (rank, id(curr), curr))
            curr = curr.next

        # 3. Process pairs in priority order (lowest merge rank first)
        while heap:
            rank, _, node = heapq.heappushpop(heap, (float("inf"), 0, None)) if not heap else heapq.heappop(heap)
            if node is None or node.next is None:
                continue

            # Verify that the candidate pair is still intact in the doubly-linked list
            pair = (node.val, node.next.val)
            if pair not in self.merges or self.merges[pair] != rank:
                continue

            # Merge nodes: Replace node value and splice out node.next
            node.val = rank
            merged_next = node.next.next
            node.next = merged_next
            if merged_next:
                merged_next.prev = node

            # Check and push newly created adjacent pairs around spliced node
            if node.prev:
                prev_pair = (node.prev.val, node.val)
                if prev_pair in self.merges:
                    heapq.heappush(heap, (self.merges[prev_pair], id(node.prev), node.prev))

            if node.next:
                next_pair = (node.val, node.next.val)
                if next_pair in self.merges:
                    heapq.heappush(heap, (self.merges[next_pair], id(node), node))

        # 4. Extract final token IDs from linked list
        ids = []
        curr = head
        while curr:
            ids.append(curr.val)
            curr = curr.next
        return ids

    def encode(self, text, allowed_special="all"):
        """Encodes string into integer token IDs handling special control tokens."""
        if allowed_special == "all":
            active_specials = self.special_tokens
        elif allowed_special == "none":
            active_specials = {}
        elif isinstance(allowed_special, set):
            active_specials = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"Invalid allowed_special configuration: {allowed_special}")

        if active_specials:
            special_pattern = "(" + "|".join(re.escape(k) for k in active_specials.keys()) + ")"
            chunks = re.split(special_pattern, text)
        else:
            chunks = [text]

        ids = []
        for chunk in chunks:
            if chunk in active_specials:
                ids.append(active_specials[chunk])
            else:
                sub_chunks = self.compiled_pattern.findall(chunk)
                for sub_chunk in sub_chunks:
                    chunk_bytes = sub_chunk.encode("utf-8")
                    ids.extend(self._encode_chunk_fast(chunk_bytes))
        return ids

    def decode(self, ids):
        """Decodes integer token sequence back into a text string."""
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token ID encountered: {idx}")

        return b"".join(part_bytes).decode("utf-8", errors="replace")

    def save(self, file_prefix):
        """Saves merge tables, patterns, and special tokens to disk."""
        model_file = file_prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("gpt4o-bpe-engine v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for k, v in self.special_tokens.items():
                f.write(f"{k} {v}\n")
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")

    def load(self, model_file):
        """Loads trained merge tables and metadata from disk."""
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
# End-to-End Verification Pipeline
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sample_corpus = """
    def execute_pipeline(x: int) -> int:
        # Testing GPT-4o 3-digit number preservation: 123456789
        value = 123 + 456789
        return value * 42

    <|fim_prefix|>def run():<|fim_suffix|> return 0<|fim_middle|>
    """

    tokenizer = OptimizedBPETokenizer()
    tokenizer.train(sample_corpus, vocab_size=320, verbose=False)

    test_input = "def run(): <|fim_prefix|>val = 123456789<|fim_suffix|>"
    encoded = tokenizer.encode(test_input, allowed_special="all")
    decoded = tokenizer.decode(encoded)

    print("Encoded IDs:", encoded)
    print("Decoded Text:", decoded)

    assert test_input == decoded, "Round-trip assertion failed!"
    print("\nRound-trip invariant verified successfully.")