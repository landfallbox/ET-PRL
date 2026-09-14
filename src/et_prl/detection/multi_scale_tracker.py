"""
多尺度分布追踪模块 - 在多个时间窗口上追踪数据分布变化
"""

from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np


class MultiScaleDistributionTracker:
    """多尺度分布追踪。"""

    def __init__(
        self,
        feature_dim: int,
        windows: Dict[str, int] | None = None,
        ema_decay: float = 0.01,
    ):
        if windows is None:
            windows = {
                "short": 100,
                "medium": 1440,
                "long": 10080,
            }

        self.feature_dim = feature_dim
        self.windows = windows
        self.ema_decay = ema_decay

        self.buffers = {name: deque(maxlen=size) for name, size in windows.items()}

        self.statistics = {
            name: {
                "mean": np.zeros(feature_dim, dtype=np.float32),
                "std": np.ones(feature_dim, dtype=np.float32),
                "cov_matrix": np.eye(feature_dim, dtype=np.float32),
            }
            for name in windows.keys()
        }

        self.sample_count = 0

    def update(self, sample: np.ndarray) -> None:
        if sample.shape[0] != self.feature_dim:
            raise ValueError(f"样本特征维度不匹配: {sample.shape[0]} != {self.feature_dim}")

        self.sample_count += 1

        for _, buffer in self.buffers.items():
            buffer.append(sample)

        if self.sample_count % 100 == 0:
            self._update_statistics()

    def _update_statistics(self) -> None:
        for window_name, buffer in self.buffers.items():
            if len(buffer) > 1:
                data = np.array(list(buffer))
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0) + 1e-8

                self.statistics[window_name]["mean"] = (
                    (1 - self.ema_decay) * self.statistics[window_name]["mean"]
                    + self.ema_decay * mean
                )
                self.statistics[window_name]["std"] = (
                    (1 - self.ema_decay) * self.statistics[window_name]["std"]
                    + self.ema_decay * std
                )

                centered = data - mean
                self.statistics[window_name]["cov_matrix"] = (
                    centered.T @ centered / max(len(data) - 1, 1)
                )

    def get_distribution(self, window: str = "long") -> Dict:
        if window not in self.windows:
            raise ValueError(f"Unknown window: {window}")
        return {
            "mean": self.statistics[window]["mean"].copy(),
            "std": self.statistics[window]["std"].copy(),
            "cov_matrix": self.statistics[window]["cov_matrix"].copy(),
        }

    def compute_kl_divergence(self, sample: np.ndarray, window: str = "long") -> float:
        dist = self.get_distribution(window)
        mean = dist["mean"]
        std = dist["std"]
        cov = dist["cov_matrix"]

        try:
            cov_inv = np.linalg.inv(cov + np.eye(self.feature_dim) * 1e-6)
            diff = sample - mean
            mahal_dist = np.sqrt(diff @ cov_inv @ diff)
        except np.linalg.LinAlgError:
            mahal_dist = np.linalg.norm((sample - mean) / (std + 1e-8))

        return float(mahal_dist)

    def detect_distribution_drift(self, threshold: float = 0.1) -> Dict[str, bool]:
        drift_detected: Dict[str, bool] = {}

        for window_name in self.windows.keys():
            if self.sample_count < 200:
                drift_detected[window_name] = False
                continue

            if window_name == "short":
                short_mean = self.statistics["short"]["mean"]
                long_mean = self.statistics["long"]["mean"]
                mean_diff = np.linalg.norm(short_mean - long_mean) / (
                    np.linalg.norm(long_mean) + 1e-8
                )

                short_std = self.statistics["short"]["std"]
                long_std = self.statistics["long"]["std"]
                std_diff = np.linalg.norm(short_std - long_std) / (
                    np.linalg.norm(long_std) + 1e-8
                )

                drift_detected[window_name] = bool((mean_diff > threshold) or (std_diff > threshold))

        return drift_detected

    def get_reference_distribution(self, window: str = "long") -> Dict:
        return self.get_distribution(window)

    def get_all_statistics(self) -> Dict:
        stats = {
            "sample_count": self.sample_count,
            "windows": {},
        }

        for window_name in self.windows.keys():
            buffer_size = len(self.buffers[window_name])
            dist = self.get_distribution(window_name)
            stats["windows"][window_name] = {
                "buffer_size": buffer_size,
                "mean": dist["mean"].copy(),
                "std": dist["std"].copy(),
            }

        return stats
