from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schemas.dataset import AppConfig


class ConfigLoader:
    """Generic YAML configuration loader.

    Responsibilities
    ----------------
    1. Read YAML
    2. Validate with a supplied Pydantic schema
    3. Return a strongly typed configuration object

    This loader is schema-agnostic and can be reused for any
    configuration (dataset, model, optimizer, training, etc.).
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)

    def _read_yaml(self) -> dict[str, Any]:
        """Read the YAML configuration file. Returns: Parsed YAML as a dictionary."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if data is None:
            raise ValueError(f"Configuration file '{self.config_path}' is empty.")

        if not isinstance(data, dict):
            raise TypeError("Configuration root must be a YAML mapping.")

        return data

    def _validate(self, data: dict[str, Any]) -> AppConfig:
        """Validate the configuration using the Pydantic schema."""
        try:
            return AppConfig.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed:\n{e}") from e

    def load(self) -> AppConfig:
        """Load and validate the configuration. Returns: AppConfig"""
        data = self._read_yaml()
        return self._validate(data)

    @classmethod
    def load_from_yaml(cls, config_path: str | Path) -> AppConfig:
        """Instantiates the loader and validates configuration in a single step.

        Parameters
        ----------
        config_path : str | Path
            The route layout pointing directly to your YAML file targets.

        Returns
        -------
        AppConfig
            A strongly typed, verified configurations object tree.
        """
        # 1. Instantiate the class dynamically using the provided path
        loader_instance = cls(config_path)
        # 2. Trigger the functional execution and validation flow
        return loader_instance.load()
