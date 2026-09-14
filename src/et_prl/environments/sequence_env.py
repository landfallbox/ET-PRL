"""
数据驱动的强化学习环境实现。

用于离线强化学习或批量数据驱动的策略优化。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .env import Env


class RewardCalculator(ABC):
    """
    奖励计算器基类。

    用户应继承此类并实现 compute 方法，
    将环境状态和动作映射到奖励值。
    """

    @abstractmethod
    def compute(self, step_index: int, action_value: float) -> Tuple[float, Dict[str, Any]]:
        """
        计算当前步骤的奖励。

        Args:
            step_index: 当前步骤索引
            action_value: 执行的动作值

        Returns:
            (奖励值, 信息字典)
        """
        raise NotImplementedError


class SequenceEnv(Env):
    """
    顺序数据环境。

    将 DataFrame 转换为可迭代的环境，支持 reset/step 接口。
    适用于离线强化学习或批量数据驱动的策略优化。
    """

    def __init__(
        self,
        data: pd.DataFrame,
        state_columns: list[str],
        reward_calculator: RewardCalculator,
    ) -> None:
        """
        初始化数据环境。

        Args:
            data: 输入数据
            state_columns: 用于构建状态的列名
            reward_calculator: 奖励计算器实例
        """
        self.data = data.reset_index(drop=True)
        self.state_columns = state_columns
        self.reward_calculator = reward_calculator
        self._index = 0

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境到初始状态。

        Returns:
            (初始观察, 空信息字典)
        """
        self._index = 0
        state = self._get_state(self._index)
        info = {}
        return state, info

    def step(self, action_value: float) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作。

        Args:
            action_value: 动作值

        Returns:
            (观察, 奖励, terminated, truncated, 信息字典)
        """
        reward, info = self.reward_calculator.compute(self._index, action_value)
        next_index = self._index + 1
        terminated = next_index >= len(self.data)

        if terminated:
            next_state = self._get_state(self._index)
        else:
            next_state = self._get_state(next_index)

        self._index = min(next_index, len(self.data) - 1)

        return next_state, reward, terminated, False, info

    def _get_state(self, index: int) -> np.ndarray:
        """获取指定索引的状态。"""
        row = self.data.iloc[index]
        state = row[self.state_columns].to_numpy(dtype=np.float32)
        return state

    @property
    def observation_size(self) -> int:
        """观察空间维度"""
        return len(self.state_columns)

    @property
    def size(self) -> int:
        """数据集大小"""
        return len(self.data)
