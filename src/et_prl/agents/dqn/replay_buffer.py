"""
强化学习经验回放缓冲区。
"""

from collections import deque
from typing import Dict, List

import numpy as np


class ReplayBuffer:
    """
    经验回放缓冲区

    存储任意字段的转移，支持灵活的字段组合。
    用户可根据具体任务添加所需字段（如 done, priority 等）。
    """

    def __init__(self, capacity: int):
        """
        初始化回放缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, **kwargs) -> None:
        """
        存储一条经验转移

        Args:
            **kwargs: 任意字段名-值对
        """
        self.buffer.append(kwargs)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        随机采样一个批量

        Args:
            batch_size: 批大小

        Returns:
            字典，键为字段名，值为对应的 numpy 数组
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲区中样本数 ({len(self.buffer)}) 小于批大小 ({batch_size})")

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sampled_transitions = [self.buffer[i] for i in indices]

        batch_dict = {}
        if sampled_transitions:
            fields = sampled_transitions[0].keys()
            for field in fields:
                values = [transition[field] for transition in sampled_transitions]
                batch_dict[field] = np.array(values)

        return batch_dict

    def is_ready(self, batch_size: int) -> bool:
        """
        检查缓冲区是否已准备好采样

        Args:
            batch_size: 所需的批大小

        Returns:
            是否可以采样
        """
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        """返回缓冲区中的样本数"""
        return len(self.buffer)

    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()

    def get_fields(self) -> List[str]:
        """
        获取缓冲区中所有转移的字段名

        Returns:
            字段名列表
        """
        if self.buffer:
            return list(self.buffer[0].keys())
        return []
