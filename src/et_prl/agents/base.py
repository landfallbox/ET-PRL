"""强化学习智能体基类定义。"""

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    """强化学习智能体基类"""

    @abstractmethod
    def learn(self) -> float:
        """
        从经验中学习

        Returns:
            损失值或学习指标
        """
        raise NotImplementedError

    @abstractmethod
    def store_transition(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        """
        存储经验转移

        Args:
            state: 当前状态
            action: 动作索引
            reward: 奖励
            next_state: 下一个状态
        """
        raise NotImplementedError
