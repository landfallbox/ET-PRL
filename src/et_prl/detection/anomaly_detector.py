"""
@Author      : landfallbox
@Date        : 2025/12/30

异常检测基类定义。
"""

from abc import ABC, abstractmethod

import numpy as np


class AnomalyDetector(ABC):
    """异常检测基类，定义统一接口。"""

    @abstractmethod
    def fit(self, x: np.ndarray) -> "AnomalyDetector":
        """训练异常检测模型"""
        raise NotImplementedError

    @abstractmethod
    def score_samples(self, x: np.ndarray) -> np.ndarray:
        """计算异常分数"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测异常标签"""
        raise NotImplementedError

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        """训练并预测异常标签"""
        return self.fit(x).predict(x)
