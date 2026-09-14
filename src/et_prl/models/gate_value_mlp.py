from __future__ import annotations

import torch
from torch import nn


class GateValueMLP(nn.Module):
    """价值触发器二分类 MLP。"""

    def __init__(self, input_dim: int, hidden_size: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim 必须大于 0，当前: {input_dim}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size 必须大于 0，当前: {hidden_size}")

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.extend(
            [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
