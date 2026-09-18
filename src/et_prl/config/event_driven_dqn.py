"""EventDrivenDQNConfig 配置（frozen dataclass schema）。值由 YAML 提供（见 configs/）。"""
from __future__ import annotations

from dataclasses import dataclass

from et_prl.config.dqn import DQNConfig
from et_prl.config.gate import GateConfig


@dataclass(frozen=True)
class EventDrivenDQNConfig(DQNConfig, GateConfig):
    pass
