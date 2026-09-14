"""
@Author      : landfallbox
@Date        : 2025/12/30

高斯噪声增强实现。
"""

from typing import Optional, Tuple

import numpy as np

from .data_augmentation import DataAugmentationStrategy


class SequenceNoiseAugmentor(DataAugmentationStrategy):
    """
    序列数据高斯噪声增强

    对指定类别的样本添加高斯噪声生成新样本，用于解决类别不平衡问题。
    """

    def __init__(
        self,
        target_class: int = 1,
        augmentation_factor: int = 1,
        noise_scale: float = 0.02,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            target_class: 目标类别（默认为1，即正样本）
            augmentation_factor: 增强倍数（生成多少倍的增强样本）
            noise_scale: 噪声强度（相对于标准差的比例）
            random_state: 随机种子，用于复现结果
        """
        self.target_class = target_class
        self.augmentation_factor = augmentation_factor
        self.noise_scale = noise_scale
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def augment(
        self, sequences: np.ndarray, labels: np.ndarray, time_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对指定类别的样本进行高斯噪声增强

        Args:
            sequences: 序列数组，shape (num_samples, sequence_length, num_features)
            labels: 标签数组，shape (num_samples,)
            time_indices: 时间索引数组，shape (num_samples,)

        Returns:
            tuple: (augmented_sequences, augmented_labels, augmented_time_indices)
                包含原始数据和增强数据
        """
        target_mask = labels == self.target_class
        target_sequences = sequences[target_mask]
        target_labels = labels[target_mask]
        target_time_indices = time_indices[target_mask]

        if len(target_sequences) == 0:
            return sequences, labels, time_indices

        augmented_sequences_list = [sequences]
        augmented_labels_list = [labels]
        augmented_time_indices_list = [time_indices]

        # 对每个目标类样本生成增强版本
        for i in range(self.augmentation_factor):
            # 噪声强度随迭代递减
            current_noise_scale = self.noise_scale / (i + 1)
            noise = np.random.normal(0, current_noise_scale, target_sequences.shape)

            # 按特征的标准差缩放噪声
            std = np.std(target_sequences, axis=0, keepdims=True)
            std[std == 0] = 1  # 避免除以0
            augmented_seq = target_sequences + noise * std

            augmented_sequences_list.append(augmented_seq)
            augmented_labels_list.append(target_labels)
            augmented_time_indices_list.append(target_time_indices)

        # 合并所有数据
        final_sequences = np.concatenate(augmented_sequences_list, axis=0)
        final_labels = np.concatenate(augmented_labels_list, axis=0)
        final_time_indices = np.concatenate(augmented_time_indices_list, axis=0)

        return final_sequences, final_labels, final_time_indices
