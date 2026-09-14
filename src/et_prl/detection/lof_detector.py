"""
@Author      : landfallbox
@Date        : 2025/12/30

LocalOutlierFactor 异常检测器实现。
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from .anomaly_detector import AnomalyDetector


class LOFDetector(AnomalyDetector):
    """LocalOutlierFactor异常检测器"""

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        n_jobs: int = -1,
        novelty: bool = True,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.novelty = novelty

        self.model: LocalOutlierFactor | None = None
        self.is_fitted = False

    def fit(self, x: np.ndarray) -> "LOFDetector":
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            n_jobs=self.n_jobs,
            novelty=self.novelty,
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
