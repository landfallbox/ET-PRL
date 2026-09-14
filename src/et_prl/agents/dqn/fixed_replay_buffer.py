"""
高性能经验回放缓冲区。

预分配内存的固定大小缓冲区实现。
"""

from __future__ import annotations

import numpy as np


class FixedReplayBuffer:
    """
    固定大小的经验回放缓冲区。

    使用预分配的 numpy 数组存储转移，适合大规模训练。
    支持固定状态维度和动作维度。
    """

    def __init__(
        self,
        capacity: int,
        state_size: int,
        state_dtype: type = np.float32,
        action_dtype: type = np.int64,
    ) -> None:
        """
        初始化回放缓冲区。

        Args:
            capacity: 缓冲区容量
            state_size: 状态维度
            state_dtype: 状态数据类型
            action_dtype: 动作数据类型
        """
        self.capacity = int(capacity)
        self.state_size = state_size

        self.state_buf = np.zeros((capacity, state_size), dtype=state_dtype)
        self.action_buf = np.zeros(capacity, dtype=action_dtype)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_size), dtype=state_dtype)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        存储一条转移。

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        idx = self.position
        self.state_buf[idx] = state
        self.action_buf[idx] = action
        self.reward_buf[idx] = reward
        self.next_state_buf[idx] = next_state
        self.done_buf[idx] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """
        随机采样一个批量。

        Args:
            batch_size: 批大小

        Returns:
            (states, actions, rewards, next_states, dones) 元组
        """
        if self.size < batch_size:
            raise ValueError(f"缓冲区中样本数 ({self.size}) 小于批大小 ({batch_size})")

        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.state_buf[indices],
            self.action_buf[indices],
            self.reward_buf[indices],
            self.next_state_buf[indices],
            self.done_buf[indices],
        )

    def is_ready(self, batch_size: int) -> bool:
        """检查缓冲区是否可采样。"""
        return self.size >= batch_size

    def __len__(self) -> int:
        """返回缓冲区中的样本数。"""
        return self.size

    def clear(self) -> None:
        """清空缓冲区。"""
        self.position = 0
        self.size = 0
