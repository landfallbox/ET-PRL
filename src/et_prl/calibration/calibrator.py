"""
Base interface for calibrated classifiers.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Calibrator(ABC):
    """校准器基类"""

    @abstractmethod
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
    ) -> "Calibrator":
        """训练并校准分类器"""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """预测概率"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测类别"""
        raise NotImplementedError
