import json
from pathlib import Path
from typing import Dict, List, Union


class CharacterTokenizer:
    """A standalone character-level tokenizer.
    Manages text tokenization and vocabulary persistence.
    """

    def __init__(self, text: str = ""):
        self.vocab: List[str] = []
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}

        if text:
            self.build_vocab(text)

    def build_vocab(self, text: str) -> None:
        """Constructs mappings from unique characters in the text corpus."""
        self.vocab = sorted(list(set(text)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, string: str) -> List[int]:
        """Maps raw text strings into arrays of integer token IDs."""
        return [self.char_to_idx[char] for char in string if char in self.char_to_idx]

    def decode(self, ids: List[int]) -> str:
        """Maps arrays of integer token IDs back into readable text strings."""
        return "".join(
            [self.idx_to_char[idx] for idx in ids if idx in self.idx_to_char]
        )

    def save_vocab(self, output_path: Union[str, Path]) -> None:
        """Serializes vocabulary metadata states to disk for consistency."""
        state = {"vocab": self.vocab}
        with open(Path(output_path), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_vocab(self, input_path: Union[str, Path]) -> None:
        """Restores a serialized vocabulary state to rebuild tokenization layouts."""
        with open(Path(input_path), "r", encoding="utf-8") as f:
            state = json.load(f)
        self.vocab = state["vocab"]
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
