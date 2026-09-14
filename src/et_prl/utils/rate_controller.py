from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RateController:
    """在线触发率闭环控制器。"""

    target_rate: float
    gain: float
    initial_bias: float = 0.0
    min_bias: float = -0.5
    max_bias: float = 0.5

    def __post_init__(self) -> None:
        self.target_rate = float(self.target_rate)
        self.gain = float(self.gain)
        self.bias = float(self.initial_bias)
        self.min_bias = float(self.min_bias)
        self.max_bias = float(self.max_bias)

    def update(self, observed_rate: float) -> float:
        """按公式 b_{t+1}=b_t+k(r_t-r*) 更新偏置。"""
        observed_rate = float(observed_rate)
        self.bias = self.bias + self.gain * (observed_rate - self.target_rate)
        self.bias = min(self.max_bias, max(self.min_bias, self.bias))
        return float(self.bias)

    def get_bias(self) -> float:
        return float(self.bias)

    def reset(self, bias: float | None = None) -> None:
        self.bias = float(self.initial_bias if bias is None else bias)
