from __future__ import annotations

from pathlib import Path

from config.schemas.dataset import AppConfig, DatasetConfig


class DatasetNotFoundError(Exception):
    """Raised when a requested dataset does not exist in the configuration."""


class DatasetManager:
    """
    Coordinates dataset preparation.

    Responsibilities
    ----------------
    - Locate dataset configuration
    - Create storage directories
    - Determine dataset paths
    - Coordinate downloading (future)
    - Coordinate checksum verification (future)
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._storage = config.environment.storage

    def list_datasets(self) -> list[str]:
        """
        Return all configured datasets.
        """
        return list(self._config.datasets.keys())

    def get_dataset(self, name: str) -> DatasetConfig:
        """
        Return the configuration for a dataset.
        """

        if name not in self._config.datasets:
            raise DatasetNotFoundError(f"Dataset '{name}' is not defined.")

        return self._config.datasets[name]

    def create_directories(self) -> None:
        """
        Create all required storage directories.
        """

        self._storage.root_dir.mkdir(parents=True, exist_ok=True)
        self._storage.raw_dir.mkdir(parents=True, exist_ok=True)
        self._storage.processed_dir.mkdir(parents=True, exist_ok=True)
        self._storage.cache_dir.mkdir(parents=True, exist_ok=True)

    def raw_dataset_path(self, name: str) -> Path:
        """
        Return the expected location of the raw dataset.
        """

        dataset = self.get_dataset(name)

        return self._storage.raw_dir / dataset.storage_mapping.raw

    def dataset_exists(self, name: str) -> bool:
        """
        Check whether the raw dataset exists.
        """

        return self.raw_dataset_path(name).exists()

    def prepare(self, name: str) -> Path:
        """
        Prepare a dataset.

        Current behaviour:
        - Create directories
        - Return dataset path

        Future behaviour:
        - Download if missing
        - Verify checksum
        """

        self.create_directories()

        dataset_path = self.raw_dataset_path(name)

        # Future:
        #
        # if not dataset_path.exists():
        #     Downloader.download(...)
        #
        # ChecksumVerifier.verify(...)

        return dataset_path
