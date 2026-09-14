"""
@Author      : landfallbox
@Date        : 2025/12/30

重采样平衡增强实现。
"""

from typing import Optional, Tuple

import numpy as np

from .data_augmentation import DataAugmentationStrategy


class SequenceResamplingAugmentor(DataAugmentationStrategy):
    """
    序列重采样增强

    通过随机欠采样多数类达到目标类别比例，解决类别不平衡问题。
    """

    def __init__(
        self, target_ratio: float = 0.5, minority_class: int = 1, random_state: Optional[int] = None
    ):
        """
        Args:
            target_ratio: 目标少数类比例
            minority_class: 少数类标签（默认为1）
            random_state: 随机种子，用于复现结果
        """
        self.target_ratio = target_ratio
        self.minority_class = minority_class
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def augment(
        self, sequences: np.ndarray, labels: np.ndarray, time_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        通过重采样达到目标类别比例

        Args:
            sequences: 序列数组
            labels: 标签数组
            time_indices: 时间索引数组

        Returns:
            tuple: (resampled_sequences, resampled_labels, resampled_time_indices)
        """
        minority_mask = labels == self.minority_class
        majority_mask = ~minority_mask

        minority_indices = np.where(minority_mask)[0]
        majority_indices = np.where(majority_mask)[0]

        num_minority = len(minority_indices)
        num_majority = len(majority_indices)

        if num_minority == 0 or num_majority == 0:
            return sequences, labels, time_indices

        # 计算需要的多数类样本数
        num_majority_target = int(num_minority / self.target_ratio) - num_minority
        num_majority_target = min(num_majority_target, num_majority)

        # 随机选择多数类样本
        selected_majority = np.random.choice(majority_indices, num_majority_target, replace=False)
        selected_indices = np.concatenate([minority_indices, selected_majority])

        return (
            sequences[selected_indices],
            labels[selected_indices],
            time_indices[selected_indices],
        )
