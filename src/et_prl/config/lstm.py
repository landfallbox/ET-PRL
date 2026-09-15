"""LSTMConfig 配置（frozen dataclass schema）。值由 YAML 提供（见 configs/）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from et_prl.config.base import BaseConfig


@dataclass(frozen=True)
class LSTMConfig(BaseConfig):
    EXPERIMENT_NAME: str
    FEATURE_COLUMNS: list
    DATA_SUBDIR: str
    WINDOW_LENGTH: int
    TARGET_COLUMN: str
    INPUT_SIZE: int
    HIDDEN_SIZES: list
    OUTPUT_SIZE: int
    BATCH_FIRST: bool
    DROPOUT: float
    LEARNING_RATE: float
    OPTIMIZER: str
    LOSS_FUNCTION: str
    BATCH_SIZE: int
    EPOCHS: int
    EARLY_STOP_PATIENCE: int
