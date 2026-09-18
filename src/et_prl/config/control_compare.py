"""ControlCompareConfig 配置（frozen dataclass schema）。值由 YAML 提供（见 configs/）。"""
from __future__ import annotations

from dataclasses import dataclass

from et_prl.config.dqn import DQNConfig
from et_prl.config.gate import GateConfig


@dataclass(frozen=True)
class ControlCompareConfig(DQNConfig, GateConfig):
    pass

    @property
    def RBC_FIXED_SETPOINT(self) -> float:
        """RBC 规则基线固定设定值，复用 DQN 目标供水温度。"""
        return self.TARGET_SUPPLY_TEMP

