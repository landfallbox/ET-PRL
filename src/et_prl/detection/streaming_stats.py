"""
流式统计模块 - 维护数据分布的在线统计信息（均值、方差等）
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class StreamingStats:
    """
    维护数据流的实时统计信息

    支持：
    - 指数移动平均（EMA）更新均值和方差
    - 滑动窗口统计
    - 特征归一化
    """

    def __init__(
        self,
        feature_dim: int,
        ema_decay: float = 0.01,
        window_size: int = 1000,
        initialize_with_data: Optional[np.ndarray] = None,
    ):
        self.feature_dim = feature_dim
        self.ema_decay = ema_decay
        self.window_size = window_size

        self.count = 0
        self.mean = np.zeros(feature_dim, dtype=np.float32)
        self.M2 = np.zeros(feature_dim, dtype=np.float32)
        self.variance = np.zeros(feature_dim, dtype=np.float32)

        self.window_buffer = deque(maxlen=window_size)

        if initialize_with_data is not None:
            self._initialize(initialize_with_data)

    def _initialize(self, data: np.ndarray) -> None:
        if data.shape[1] != self.feature_dim:
            raise ValueError(f"初始化数据特征维度不匹配: {data.shape[1]} != {self.feature_dim}")

        self.mean = np.mean(data, axis=0, dtype=np.float32)
        self.variance = np.var(data, axis=0, dtype=np.float32)
        self.M2 = self.variance * data.shape[0]
        self.count = data.shape[0]

        for sample in data:
            self.window_buffer.append(sample)

    def update(self, sample: np.ndarray) -> None:
        if sample.shape[0] != self.feature_dim:
            raise ValueError(f"样本特征维度不匹配: {sample.shape[0]} != {self.feature_dim}")

        self.count += 1
        delta = sample - self.mean
        self.mean += self.ema_decay * delta
        delta2 = sample - self.mean
        self.M2 += delta * delta2

        if self.count > 1:
            self.variance = self.M2 / self.count
        else:
            self.variance = np.zeros(self.feature_dim, dtype=np.float32)

        self.window_buffer.append(sample)

    def get_mean(self) -> np.ndarray:
        return np.asarray(self.mean, dtype=np.float32).copy()

    def get_variance(self) -> np.ndarray:
        return np.asarray(self.variance, dtype=np.float32).copy()

    def get_std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 1e-8))

    def get_local_mean(self) -> np.ndarray:
        if len(self.window_buffer) == 0:
            return np.asarray(self.mean, dtype=np.float32).copy()
        return np.mean(list(self.window_buffer), axis=0)

    def get_local_std(self) -> np.ndarray:
        if len(self.window_buffer) == 0:
            return self.get_std()
        return np.std(list(self.window_buffer), axis=0, dtype=np.float32)

    def normalize(self, sample: np.ndarray, use_global: bool = True) -> np.ndarray:
        if use_global:
            mean = self.mean
            std = self.get_std()
        else:
            mean = self.get_local_mean()
            std = self.get_local_std()

        return (sample - mean) / (std + 1e-8)

    def denormalize(self, sample: np.ndarray, use_global: bool = True) -> np.ndarray:
        if use_global:
            mean = self.mean
            std = self.get_std()
        else:
            mean = self.get_local_mean()
            std = self.get_local_std()

        return sample * (std + 1e-8) + mean

    def get_statistics(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean.copy(),
            "variance": self.variance.copy(),
            "std": self.get_std(),
            "local_mean": self.get_local_mean(),
            "local_std": self.get_local_std(),
            "window_size": len(self.window_buffer),
        }
