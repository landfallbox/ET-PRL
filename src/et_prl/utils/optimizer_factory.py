"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 优化器工厂
"""

import torch
import torch.nn as nn


def create_optimizer(
    model: nn.Module, optimizer_type: str, learning_rate: float, **kwargs
) -> torch.optim.Optimizer:
    """
    创建优化器
    参数：
        model: 模型
        optimizer_type: 优化器类型（adam, sgd, rmsprop）
        learning_rate: 学习率
        **kwargs: 其他优化器参数（如momentum、weight_decay等）
    返回：
        优化器实例
    异常：
        ValueError: 当优化器类型不支持时抛出
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}，支持的类型: adam, sgd, rmsprop")
