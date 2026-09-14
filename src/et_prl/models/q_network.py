"""
DQN 神经网络实现。

支持灵活的隐层配置。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q网络：多层 MLP 架构。

    将状态映射到各离散动作的Q值。
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        """
        初始化Q网络。

        Args:
            state_size: 状态空间维度
            action_size: 动作空间大小（离散）
            hidden_sizes: 隐层尺寸列表。默认为 [128]
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128]

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes

        layers: list[nn.Module] = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 状态张量，形状为 (batch_size, state_size)

        Returns:
            Q值张量，形状为 (batch_size, action_size)
        """
        return self.net(x)

    def get_model_info(self) -> dict:
        """获取模型信息。"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "QNetwork_MLP",
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_sizes": self.hidden_sizes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
