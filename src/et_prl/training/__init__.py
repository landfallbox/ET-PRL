"""训练框架。"""

from .base import Trainer
from .lstm_trainer import LSTMTrainer

__all__ = ["Trainer", "LSTMTrainer"]
