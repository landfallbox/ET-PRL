"""
@Author      : landfallbox
@Date        : 2025/12/30

IsolationForest 异常检测器实现。
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from .anomaly_detector import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    """IsolationForest异常检测器"""

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model: IsolationForest | None = None
        self.is_fitted = False

    def fit(self, x: np.ndarray) -> "IsolationForestDetector":
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(x)
        self.is_fitted = True
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return -self.model.score_samples(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(x)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return -self.model.decision_function(x)

