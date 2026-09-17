"""DQN 智能体实现。"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from ..base import Agent
from .replay_buffer import ReplayBuffer
from et_prl.models import QNetwork
from et_prl.utils import create_optimizer


class DQNAgent(Agent):
    """
    DQN 智能体：用于离散动作选择和经验学习。

    使用 QNetwork 作为策略网络，ε-贪心探索，经验回放训练。
    """

    def __init__(
        self,
        state_size: int,
        action_space: np.ndarray,
        hidden_sizes: list[int],
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        memory_capacity: int,
        batch_size: int,
        target_update_freq: int,
        device: torch.device,
    ) -> None:
        """
        初始化 DQN 智能体。

        Args:
            state_size: 状态空间维度
            action_space: 实际动作值数组（离散动作）
            hidden_sizes: 策略网络隐层尺寸列表
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_min: 最小探索率
            epsilon_decay: 每次学习步的探索率衰减系数
            memory_capacity: 经验回放缓冲区容量
            batch_size: 训练批大小
            target_update_freq: 目标网络更新频率（学习步数）
            device: PyTorch 设备
        """
        hidden_sizes = hidden_sizes or [128]
        self.state_size = state_size
        self.action_size = len(action_space)
        self.action_space = np.asarray(action_space)
        self.gamma = gamma
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.device = device

        net_kwargs = {
            "state_size": state_size,
            "action_size": self.action_size,
            "hidden_sizes": list(hidden_sizes),
        }
        self.policy_net = QNetwork(**net_kwargs).to(device)
        self.target_net = QNetwork(**net_kwargs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = create_optimizer(self.policy_net, "adam", learning_rate)
        self.loss_func = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(int(memory_capacity))
        self.learn_counter = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """使用 ε-贪心策略选择动作索引。"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def get_action_value(self, action_idx: int) -> float:
        """从动作索引获取实际动作值。"""
        return float(self.action_space[action_idx])

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ) -> None:
        """存储经验转移到回放缓冲区。"""
        self.replay_buffer.push(state=state, action=action, reward=reward, next_state=next_state)

    def learn(self) -> float:
        """从回放缓冲区采样并更新策略网络，返回损失值。"""
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_states = batch["next_state"]

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + self.gamma * next_q_values

        loss = self.loss_func(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_counter += 1
        if self.learn_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())
