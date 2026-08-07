import logging

import torch

from src.indra_llm.config.schemas.dataset import AppConfig
from src.indra_llm.tokenizer.character_level_tokenizer import CharacterTokenizer

logger = logging.getLogger(__name__)


def run_preprocessing(
    config: AppConfig, dataset_key: str = "tiny_shakespeare_raw"
) -> None:
    """Orchestrates ingestion, tokenization, serialization, and sequential splits."""
    ds_config = config.datasets[dataset_key]

    raw_path = config.environment.storage.raw_dir / ds_config.source.filename
    processed_dir = (
        config.environment.storage.processed_dir / ds_config.storage_mapping.processed
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read clean input text
    logger.info(f"Loading raw file for tokenization from: {raw_path}")
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Fit Tokenizer configuration and save metadata state
    logger.info("Building dataset tokenizer map...")
    tokenizer = CharacterTokenizer(text=text)
    vocab_path = processed_dir / "vocab.json"
    tokenizer.save_vocab(vocab_path)
    logger.info(f"Saved serialization metadata state to: {vocab_path}")

    # 3. Complete Text Tokenization translation
    token_ids = tokenizer.encode(text)

    # 4. Generate Causal-Safe Sequential Tensors
    # Splitting index targets chronologically instead of shuffling to protect text flow
    split_ratio = (
        config.datasets[dataset_key].source.train_ratio
        if hasattr(ds_config.source, "train_ratio")
        else 0.90
    )
    split_idx = int(len(token_ids) * split_ratio)

    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    # 5. Convert lists directly into continuous PyTorch long tensors
    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    val_tensor = torch.tensor(val_ids, dtype=torch.long)

    # Save finalized token binaries to disk (.pt format)
    torch.save(train_tensor, processed_dir / "train.pt")
    torch.save(val_tensor, processed_dir / "val.pt")

    logger.info(
        f"Preprocessing stage absolute finish! Encoded {len(token_ids):,} total items."
    )
    logger.info(
        f"Train tensor items: {train_tensor.shape[0]:,}, "
        f"Val tensor items: {val_tensor.shape[0]:,}"
    )
