"""
通用强化学习环境基类。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class Env(ABC):
    """
    强化学习环境基类。

    遵循 Gym/Gymnasium 的 reset/step 接口设计。
    """

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境到初始状态。

        Returns:
            (初始观察, 信息字典)
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action_value: float) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作。

        Args:
            action_value: 动作值

        Returns:
            (观察, 奖励, terminated, truncated, 信息字典)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_size(self) -> int:
        """观察空间维度"""
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        """环境数据量或时间步长"""
        raise NotImplementedError
