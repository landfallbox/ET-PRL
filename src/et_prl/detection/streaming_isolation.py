"""
流式异常检测模块 - 基于参考集的在线孤立度计算
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.distance import euclidean


class StreamingIsolationDepth:
    """流式异常检测（参考集孤立度）。"""

    def __init__(
        self,
        n_reference_samples: int = 500,
        update_freq: int = 100,
        contamination: float = 0.2,
    ):
        self.n_reference_samples = n_reference_samples
        self.update_freq = update_freq
        self.contamination = contamination

        self.reference_buffer = deque(maxlen=n_reference_samples)
        self.reference_array = None

        self.short_buffer = deque(maxlen=100)
        self.medium_buffer = deque(maxlen=1440)
        self.long_buffer = deque(maxlen=10080)

        self.distance_stats = {
            "short": {"mean": 0.0, "std": 1.0},
            "medium": {"mean": 0.0, "std": 1.0},
            "long": {"mean": 0.0, "std": 1.0},
        }

        self.sample_count = 0
        self.covariance_matrix = None

    def _get_window_array(self, window: str) -> np.ndarray | None:
        window_buffer = {
            "short": self.short_buffer,
            "medium": self.medium_buffer,
            "long": self.long_buffer,
        }.get(window)

        if window_buffer is None:
            raise ValueError(f"Unknown window: {window}")

        if len(window_buffer) > 0:
            return np.asarray(list(window_buffer), dtype=np.float32)
        if self.reference_array is not None and len(self.reference_array) > 0:
            return np.asarray(self.reference_array, dtype=np.float32)
        return None

    @staticmethod
    def _normalize_distances(distances: np.ndarray, feature_dim: int) -> np.ndarray:
        dim_scale = max(np.sqrt(float(feature_dim)), 1.0)
        return distances / dim_scale

    def _initialize_reference_set(self, initial_data: np.ndarray) -> None:
        indices = np.random.choice(
            len(initial_data),
            size=min(self.n_reference_samples, len(initial_data)),
            replace=False,
        )
        for idx in indices:
            self.reference_buffer.append(initial_data[idx])

        self.reference_array = np.array(list(self.reference_buffer))
        self._update_distance_statistics()

    def _update_reference_set(self, sample: np.ndarray) -> None:
        self.reference_buffer.append(sample)

        if self.sample_count % self.update_freq == 0:
            self.reference_array = np.array(list(self.reference_buffer))
            self._update_distance_statistics()

    def _update_distance_statistics(self) -> None:
        for window_name in ("short", "medium", "long"):
            window_array = self._get_window_array(window_name)
            if window_array is None or len(window_array) < 2:
                continue

            max_samples = min(128, len(window_array))
            sample_indices = np.linspace(0, len(window_array) - 1, num=max_samples, dtype=int)
            sample_subset = window_array[sample_indices]
            feature_dim = sample_subset.shape[1]

            distances = []
            for i in range(max_samples):
                diff = sample_subset[i + 1 :] - sample_subset[i]
                if diff.size == 0:
                    continue
                pairwise_distances = np.linalg.norm(diff, axis=1)
                distances.extend(self._normalize_distances(pairwise_distances, feature_dim).tolist())

            if distances:
                distance_array = np.asarray(distances, dtype=np.float32)
                self.distance_stats[window_name]["mean"] = float(np.mean(distance_array))
                self.distance_stats[window_name]["std"] = float(np.std(distance_array)) + 1e-8

    def _compute_isolation_distance(self, sample: np.ndarray, window: str = "long") -> float:
        window_array = self._get_window_array(window)
        if window_array is None or len(window_array) == 0:
            return 0.0

        feature_dim = int(window_array.shape[1]) if window_array.ndim == 2 else 1
        distances = np.linalg.norm(window_array - sample, axis=1)

        normalized_distances = self._normalize_distances(distances, feature_dim)
        if normalized_distances.size == 0:
            return 0.0

        k = min(max(5, int(np.sqrt(normalized_distances.size))), 32, normalized_distances.size)
        nearest_distances = np.partition(normalized_distances, k - 1)[:k]

        return float(np.mean(nearest_distances))

    def update(self, sample: np.ndarray) -> None:
        self.sample_count += 1
        self._update_reference_set(sample)

        self.short_buffer.append(sample)
        self.medium_buffer.append(sample)
        self.long_buffer.append(sample)

    def score(self, sample: np.ndarray, window: str = "long") -> float:
        if self.reference_array is None or len(self.reference_array) == 0:
            return 0.0

        distance = self._compute_isolation_distance(sample, window)

        stats = self.distance_stats[window]
        if stats["std"] > 1e-8:
            normalized_score = (distance - stats["mean"]) / stats["std"]
            anomaly_score = 1.0 / (1.0 + np.exp(-normalized_score))
        else:
            anomaly_score = 0.5

        return float(np.clip(anomaly_score, 0.0, 1.0))

    def score_batch(self, samples: np.ndarray, window: str = "long") -> np.ndarray:
        return np.array([self.score(sample, window) for sample in samples])

    def get_statistics(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "reference_set_size": len(self.reference_buffer),
            "distance_stats": self.distance_stats,
            "short_buffer_size": len(self.short_buffer),
            "medium_buffer_size": len(self.medium_buffer),
            "long_buffer_size": len(self.long_buffer),
        }
