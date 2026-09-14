"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 损失函数工厂
"""

import torch.nn as nn


def create_loss_fn(loss_type: str) -> nn.Module:
    """
    创建损失函数
    参数：
        loss_type: 损失函数类型（mse, mae, ce）
    返回：
        损失函数实例
    异常：
        ValueError: 当损失函数类型不支持时抛出
    """
    loss_type = loss_type.lower()
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}，支持的类型: mse, mae, ce")
