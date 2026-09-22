"""DQNConfig 配置（frozen dataclass schema）。值由 YAML 提供（见 configs/）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from et_prl.config.base import BaseConfig


@dataclass(frozen=True)
class DQNConfig(BaseConfig):
    EXPERIMENT_NAME: str
    DATA_SUBDIR: str
    STATE_COLUMNS: list
    TRAIN_FILENAME: str
    VAL_FILENAME: str
    TEST_FILENAME: str
    STATE_SIZE: int
    HIDDEN_SIZES: list
    LEARNING_RATE: float
    GAMMA: float
    EPSILON_START: float
    EPSILON_MIN: float
    EPSILON_DECAY: float
    MEMORY_CAPACITY: int
    BATCH_SIZE: int
    TARGET_UPDATE_FREQ: int
    NUM_EPISODES: int
    EARLY_STOPPING_PATIENCE: int
    VAL_INTERVAL: int
    REWARD_WEIGHT_EFFICIENCY: float
    REWARD_WEIGHT_COMFORT: float
    TARGET_SUPPLY_TEMP: float
    CHILLER_SUPPLY_TEMP_REF: float
    COMFORT_SIGMA: float
    CHILLER_CAPACITY: int
    CHILLER_REF_POWER: int
    CHILLER_F_CHW: int
    CHILLER_CP: float
    CHILLER_WATER_DENSITY: int
    CHILLER_HIGH_THRESHOLD: int
    CHILLER_MEDIUM_THRESHOLD: int
    CHILLER_LOW_THRESHOLD: int

    @property
    def ACTION_SPACE_PATH(self) -> Path:
        return self.DATA_ROOT / "action_all.npy"

    @property
    def COEFF_DATE_PATH(self) -> Path:
        return self.DATA_ROOT / "coeff_date.npy"

    def validate(self) -> None:
        total = self.REWARD_WEIGHT_EFFICIENCY + self.REWARD_WEIGHT_COMFORT
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Reward weights must sum to 1.0, got {total}")
        if not (0 <= self.EPSILON_MIN <= self.EPSILON_START <= 1.0):
            raise ValueError(
                "Epsilon values must satisfy: 0 <= EPSILON_MIN <= EPSILON_START <= 1.0"
            )
