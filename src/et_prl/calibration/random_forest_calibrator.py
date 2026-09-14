from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator

from .calibrator import Calibrator


class RandomForestCalibrator(Calibrator):
    """RandomForest + Platt Scaling校准器"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        class_weight: Optional[str] = "balanced",
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        calibration_method: str = "sigmoid",
        cv: int = 5,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.calibration_method = calibration_method
        self.cv = cv

        self.base_clf = None
        self.calibrated_clf = None
        self.is_fitted = False

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
    ) -> "RandomForestCalibrator":
        self.base_clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.base_clf.fit(x_train, y_train)

        frozen_clf = FrozenEstimator(self.base_clf)
        self.calibrated_clf = CalibratedClassifierCV(
            frozen_clf, method=self.calibration_method, cv=self.cv if x_cal is None else None
        )

        if x_cal is not None and y_cal is not None:
            self.calibrated_clf.fit(x_cal, y_cal)
        else:
            self.calibrated_clf.fit(x_train, y_train)

        self.is_fitted = True
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.calibrated_clf.predict_proba(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.calibrated_clf.predict(x)

    def predict_proba_positive(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return proba[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        """获取特征重要性"""
        if not self.is_fitted or self.base_clf is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.base_clf.feature_importances_

