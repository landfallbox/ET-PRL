"""
@Author      : landfallbox
@Date        : 2025/12/30

数据增强基类定义。
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class DataAugmentationStrategy(ABC):
    """数据增强策略基类"""

    @abstractmethod
    def augment(
        self, sequences: np.ndarray, labels: np.ndarray, time_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对数据进行增强

        Args:
            sequences: 序列数组，shape (num_samples, sequence_length, num_features)
            labels: 标签数组，shape (num_samples,)
            time_indices: 时间索引数组，shape (num_samples,)

        Returns:
            tuple: (augmented_sequences, augmented_labels, augmented_time_indices)
        """
        raise NotImplementedError

    @staticmethod
    def compute_class_distribution(labels: np.ndarray) -> dict:
        """
        计算类别分布统计

        Args:
            labels: 标签数组

        Returns:
            dict: 包含各类别的数量和比例
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        distribution = {}
        for label, count in zip(unique_labels, counts):
            distribution[int(label)] = {"count": int(count), "ratio": float(count / total)}

        return distribution
