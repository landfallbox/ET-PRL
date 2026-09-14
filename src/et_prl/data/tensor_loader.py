"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 张量加载和转换工具
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DataLoaderConfig:
    """数据加载器配置类"""

    batch_size: int
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False


def load_csv_to_tensor(
    file_path: Path, target_column_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    加载CSV文件并转换为PyTorch张量
    参数：
        file_path: CSV文件路径
        target_column_name: 目标列名（完整列名）
    返回：
        (features_tensor, targets_tensor)
        features_tensor: shape (num_samples, num_features)，dtype=float32
        targets_tensor: shape (num_samples, 1)，dtype=float32
    异常：
        FileNotFoundError: 当CSV文件不存在时抛出
        ValueError: 当目标列不存在时抛出
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV文件不存在: {file_path}")

    df = pd.read_csv(file_path)

    # 检查 DataFrame 是否为空
    if df.empty:
        raise ValueError(f"CSV文件为空: {file_path}")

    # 分离特征列和目标列
    target_col = target_column_name
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 在CSV中不存在，可用列: {df.columns.tolist()}")
    feature_cols = [col for col in df.columns if col != target_col]
    # 提取特征和目标
    features = df[feature_cols].values.astype(np.float32)
    targets = df[[target_col]].values.astype(np.float32)
    # 转换为张量
    features_tensor = torch.from_numpy(features)
    targets_tensor = torch.from_numpy(targets)
    return features_tensor, targets_tensor


def reshape_to_sequence_format(
    features: torch.Tensor, seq_length: int, input_size: int
) -> torch.Tensor:
    """
    将特征张量reshape为RNN序列格式（用于LSTM、GRU等循环神经网络）
    参数：
        features: shape (num_samples, seq_length * input_size) 的扁平特征张量
        seq_length: 序列长度
        input_size: 输入特征维度
    返回：
        shape (num_samples, seq_length, input_size) 的序列格式张量
    异常：
        ValueError: 当特征总维度与seq_length * input_size不匹配时抛出
    """
    num_samples = features.shape[0]
    expected_features = seq_length * input_size
    if features.shape[1] != expected_features:
        raise ValueError(f"特征维度不匹配。期望 {expected_features}，实际 {features.shape[1]}")
    reshaped = features.reshape(num_samples, seq_length, input_size)
    return reshaped


def build_sliding_window_sequences(features: torch.Tensor, seq_length: int) -> torch.Tensor:
    """
    基于单时刻特征构建滑动窗口序列，时间顺序为当前时刻到历史时刻

    参数：
        features: shape (num_rows, input_size) 的单时刻特征张量
        seq_length: 序列长度

    返回：
        shape (num_rows - seq_length + 1, seq_length, input_size) 的序列张量
    """
    if seq_length < 1:
        raise ValueError(f"seq_length 必须大于等于 1，当前值为 {seq_length}")

    if features.ndim != 2:
        raise ValueError(f"features 必须是二维张量，当前维度为 {features.ndim}")

    num_rows = features.shape[0]
    if num_rows < seq_length:
        raise ValueError(
            f"样本数量不足。当前行数={num_rows}，序列长度={seq_length}，至少需要 {seq_length} 行"
        )

    num_samples = num_rows - seq_length + 1
    start_indices = torch.arange(num_samples, device=features.device).unsqueeze(1)
    time_offsets = torch.arange(seq_length - 1, -1, -1, device=features.device).unsqueeze(0)
    time_indices = start_indices + time_offsets
    return features[time_indices].contiguous()


def load_csv_to_sequence_tensor(
    file_path: Path,
    target_column_name: str,
    seq_length: int,
    input_size: int,
    input_format: Literal["flattened", "raw"] = "flattened",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    直接从CSV加载并转换为RNN序列格式张量

    参数：
        file_path: CSV文件路径
        target_column_name: 目标列名（完整列名）
        seq_length: 序列长度
        input_size: 输入特征维度
        input_format: 输入特征格式
            - "flattened": 每行特征已展平为 seq_length * input_size
            - "raw": 每行特征为单时刻 input_size，将在函数内构建滑动窗口

    返回：
        (sequence_features, targets)
        sequence_features: shape (num_samples, seq_length, input_size)
        targets: shape (num_samples, 1)
    """
    if seq_length < 1:
        raise ValueError(f"seq_length 必须大于等于 1，当前值为 {seq_length}")

    if input_size < 1:
        raise ValueError(f"input_size 必须大于等于 1，当前值为 {input_size}")

    if input_format not in {"flattened", "raw"}:
        raise ValueError(f"input_format 仅支持 'flattened' 或 'raw'，当前值为 {input_format}")

    features, targets = load_csv_to_tensor(file_path=file_path, target_column_name=target_column_name)

    if input_format == "flattened":
        sequence_features = reshape_to_sequence_format(
            features=features,
            seq_length=seq_length,
            input_size=input_size,
        )
        return sequence_features, targets

    if features.shape[1] != input_size:
        raise ValueError(
            "raw 格式下特征维度不匹配。"
            f"期望 input_size={input_size}，实际特征维度={features.shape[1]}"
        )

    sequence_features = build_sliding_window_sequences(features, seq_length)
    aligned_targets = targets[seq_length - 1 :]
    return sequence_features, aligned_targets


def create_data_loaders(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    batch_size: Optional[int] = None,
    shuffle_train: bool = True,
    config: Optional[DataLoaderConfig] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器

    参数：
        train_features/targets: 训练集张量
        val_features/targets: 验证集张量
        test_features/targets: 测试集张量
        batch_size: 批大小（向后兼容参数，优先级低于 config）
        shuffle_train: 是否对训练集打乱（向后兼容参数，优先级低于 config）
        config: DataLoaderConfig 配置对象（推荐使用）

    返回：
        (train_loader, val_loader, test_loader)
    """
    # 如果提供了 config，使用 config；否则使用传统参数
    if config is None:
        if batch_size is None:
            raise ValueError("必须提供 batch_size 或 config 参数")
        config = DataLoaderConfig(batch_size=batch_size, shuffle_train=shuffle_train)

    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    test_dataset = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_val,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_test,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader
