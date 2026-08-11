import logging
from pathlib import Path

from src.transformer.config.loader import ConfigLoader
from src.transformer.data.dataset import GPTDataBatcher
from src.transformer.data.downloader import DatasetDownloader
from src.transformer.preprocessing.preprocessing import run_preprocessing

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # 1. Global Setup Configuration loading
    config = ConfigLoader.load_from_yaml(Path("configs/dataset.yaml"))
    dataset_key = "tiny_shakespeare"
    ds_config = config.datasets[dataset_key]

    raw_file_path = config.environment.storage.raw_dir / ds_config.source.filename
    processed_dir = (
        config.environment.storage.processed_dir / ds_config.storage_mapping.processed
    )

    # 2. Stage 1: Download raw text files securely
    # 1. Instantiate the downloader safely using its valid parameter signature
    downloader = DatasetDownloader()
    # 2. Trigger the download function by passing its expected configuration fields
    downloader.download(
        url=str(ds_config.source.url), destination=raw_file_path, overwrite=False
    )

    # 3. Stage 2: Execute Tokenization & Serialization Processing
    # This runs once to translate the text file into reusable binary shards (.pt files)
    run_preprocessing(config=config, dataset_key=dataset_key)

    # 4. Stage 3: Operational Model Ingestion Training Loops
    # The training loop interacts purely with pre-tokenized binary tensors on disk
    logger.info("Initializing runtime PyTorch Data Batchers...")
    train_batcher = GPTDataBatcher(
        tensor_path=processed_dir / "train.pt", block_size=8, batch_size=4
    )

    # Generate an active input (X) and shifted label target matrix (Y)
    x_batch, y_batch = train_batcher.get_batch()

    logger.info("Pipeline Fully Locked and Loaded!")
    logger.info(f"Batched Tensor Inputs Shape  (X): {x_batch.shape}")
    logger.info(f"Batched Tensor Targets Shape (Y): {y_batch.shape}")


if __name__ == "__main__":
    main()
