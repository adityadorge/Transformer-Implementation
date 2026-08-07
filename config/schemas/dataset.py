from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class EnvironmentMode(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SourceType(str, Enum):
    HTTP = "http"
    LOCAL = "local"


class TokenizerType(str, Enum):
    CHARACTER = "character"
    BPE = "bpe"
    WORDPIECE = "wordpiece"


class ChecksumAlgorithm(str, Enum):
    SHA256 = "sha256"


class StorageConfig(BaseModel):
    """Global storage locations."""

    model_config = ConfigDict(extra="forbid")
    root_dir: Path
    raw_dir: Path
    processed_dir: Path
    cache_dir: Path

    @field_validator("root_dir", "raw_dir", "processed_dir", "cache_dir", mode="before")
    @classmethod
    def resolve_absolute_paths(cls, value: Any) -> Path:
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                return (PROJECT_ROOT / path).resolve()
            return path
        return Path(value)


class EnvironmentConfig(BaseModel):
    """Application runtime configuration."""

    model_config = ConfigDict(extra="forbid")
    mode: EnvironmentMode
    storage: StorageConfig


class MetadataConfig(BaseModel):
    """Human-readable dataset information."""

    model_config = ConfigDict(extra="forbid")
    description: str
    license: str


class ChecksumConfig(BaseModel):
    """Checksum verification."""

    model_config = ConfigDict(extra="forbid")
    algorithm: ChecksumAlgorithm
    value: str


class SourceConfig(BaseModel):
    """Dataset source information."""

    model_config = ConfigDict(extra="forbid")
    type: SourceType
    url: HttpUrl
    filename: str
    format: str
    encoding: str
    checksum: ChecksumConfig


class DatasetStorageConfig(BaseModel):
    """Dataset-specific storage paths."""

    model_config = ConfigDict(extra="forbid")
    raw: str
    processed: str
    cache: str


class TokenizerConfig(BaseModel):
    """Tokenizer configuration."""

    model_config = ConfigDict(extra="forbid")
    type: TokenizerType
    special_tokens: dict[str, str] = Field(default_factory=dict)


class CacheConfig(BaseModel):
    """Dataset cache configuration."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool
    compress: bool


class DatasetConfig(BaseModel):
    """Configuration for one dataset."""

    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    metadata: MetadataConfig
    source: SourceConfig
    storage_mapping: DatasetStorageConfig
    tokenizer: TokenizerConfig
    cache: CacheConfig


class AppConfig(BaseModel):
    """Root application configuration."""

    model_config = ConfigDict(extra="forbid")
    config_version: str
    environment: EnvironmentConfig
    datasets: dict[str, DatasetConfig]
