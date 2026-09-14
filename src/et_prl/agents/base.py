"""
@Author      : landfallbox
@Date        : 2026/02/11

强化学习智能体基类定义。
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Agent(ABC):
    """强化学习智能体基类"""

    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> float:
        """
        选择动作

        Args:
            state: 当前状态
            training: 是否处于训练模式

        Returns:
            动作值
        """
        raise NotImplementedError

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

    @abstractmethod
    def save_checkpoint(self, filepath: str, episode: int = 0, **kwargs) -> None:
        """
        保存检查点

        Args:
            filepath: 保存路径
            episode: 当前轮数
            **kwargs: 其他元数据
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_checkpoint(
        filepath: str,
        device,
        action_space: np.ndarray | None = None,
        policy_net_class: type | None = None,
        policy_net_kwargs: dict | None = None,
    ) -> Tuple["Agent", dict]:
        """
        加载检查点

        Args:
            filepath: 检查点文件路径
            device: 设备对象
            action_space: 动作空间数组（可选）
            policy_net_class: 自定义网络类（可选）
            policy_net_kwargs: 自定义网络参数（可选）

        Returns:
            (智能体实例, 检查点字典)
        """
        raise NotImplementedError
