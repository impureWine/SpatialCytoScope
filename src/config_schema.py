"""Validated configuration and hardware selection for SpatialCytoScope."""

from pathlib import Path
from typing import Annotated, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class StrictConfig(BaseModel):
    """Reject misspelled configuration keys and nonfinite numbers."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PathsConfig(StrictConfig):
    """Locations resolved relative to the project root."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    results_dir: Path = Path("results")


class QCConfig(StrictConfig):
    """Quality control thresholds for subsequent preprocessing."""

    min_counts: PositiveInt = 100
    min_genes: PositiveInt = 150
    max_mt_pct: float = Field(default=25.0, ge=0, le=100)
    n_top_genes: PositiveInt = 800


class SpatialConfig(StrictConfig):
    """Spatial graph construction parameters."""

    n_neighbors: PositiveInt = 6
    sigma: float = Field(default=50.0, gt=0)


class ModelConfig(StrictConfig):
    """Pure PyTorch model and objective parameters."""

    hidden_dim: PositiveInt = 128
    latent_dim: PositiveInt = 30
    heads: PositiveInt = 4
    lr: float = Field(default=0.001, gt=0)
    epochs: PositiveInt = 60
    lambda_task: float = Field(default=1.0, ge=0)
    lambda_smooth: float = Field(default=0.3, ge=0)

    @model_validator(mode="after")
    def validate_heads(self) -> "ModelConfig":
        """Ensure the hidden representation splits into attention heads.

        Returns:
            Validated model configuration.

        Raises:
            ValueError: If hidden_dim cannot be divided by heads.
        """
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        return self


class AppConfig(StrictConfig):
    """Top-level application configuration."""

    seed: int = Field(default=42, strict=True, ge=0, le=2**32 - 1)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_config(path: Path | str = PROJECT_ROOT / "config/config.yaml") -> AppConfig:
    """Load YAML and resolve relative paths against the project root.

    Args:
        path: YAML configuration file to validate.

    Returns:
        Validated application configuration with absolute data paths.

    Raises:
        ValueError: If the YAML document is not a mapping.
    """
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = AppConfig.model_validate(document)
    for name in PathsConfig.model_fields:
        value = getattr(config.paths, name)
        setattr(config.paths, name, (PROJECT_ROOT / value).resolve())
    return config


def detect_device() -> Literal["cuda", "mps", "cpu"]:
    """Select the best available PyTorch device.

    Returns:
        CUDA, Apple Metal, or CPU device name in preference order.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
