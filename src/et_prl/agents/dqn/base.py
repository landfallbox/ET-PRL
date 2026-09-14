"""
DQN 智能体实现。
"""

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..base import Agent
from .replay_buffer import ReplayBuffer
from et_prl.models import QNetwork


class DQNAgent(Agent):
    """
    DQN 智能体：用于动作选择和经验学习

    支持自定义网络注入：用户可以创建继承 nn.Module 的自定义网络类，
    并通过 policy_net_class 参数传入。自定义网络应实现 forward() 和 get_model_info() 方法。
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_space: np.ndarray,
        device: torch.device,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_capacity: int = 10000,
        batch_size: int = 32,
        target_update: int = 10,
        hidden_size: int = 128,
        policy_net_class: Optional[type] = None,
        policy_net_kwargs: Optional[dict] = None,
    ) -> None:
        """
        初始化 DQN 智能体

        Args:
            state_size: 状态空间维度
            action_size: 离散动作数
            action_space: 实际动作值数组
            device: PyTorch 设备（必需，由用户传入）
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_min: 最小探索率
            epsilon_decay: epsilon 衰减率
            memory_capacity: 重放缓冲区大小
            batch_size: 批大小
            target_update: 目标网络更新频率（学习步数）
            hidden_size: 隐层大小（仅在使用默认网络时有效）
            policy_net_class: 自定义网络类（继承 nn.Module），默认为 QNetwork
            policy_net_kwargs: 传给自定义网络的额外参数字典
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        self.hidden_size = hidden_size
        self.policy_net_class = policy_net_class or QNetwork
        self.policy_net_kwargs = policy_net_kwargs or {}

        # 创建网络实例
        self._create_networks()

        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(memory_capacity)
        self.learn_counter = 0

    def _create_networks(self) -> None:
        """创建策略网络和目标网络"""
        # 构建网络初始化参数
        net_kwargs: dict[str, Any] = {
            "state_size": self.state_size,
            "action_size": self.action_size,
        }

        extra_kwargs = dict(self.policy_net_kwargs)

        # 对默认 QNetwork 统一转换为 hidden_sizes 参数
        if self.policy_net_class is QNetwork:
            if "hidden_sizes" not in extra_kwargs:
                if "hidden_size" in extra_kwargs:
                    extra_kwargs["hidden_sizes"] = [int(extra_kwargs.pop("hidden_size"))]
                else:
                    extra_kwargs["hidden_sizes"] = [self.hidden_size]
        else:
            if "hidden_size" not in extra_kwargs and "hidden_sizes" not in extra_kwargs:
                extra_kwargs["hidden_size"] = self.hidden_size

        # 合并用户提供的额外参数
        net_kwargs.update(extra_kwargs)

        # 创建策略网络和目标网络
        self.policy_net = self.policy_net_class(**net_kwargs).to(self.device)
        self.target_net = self.policy_net_class(**net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def act(self, state: np.ndarray, training: bool = True) -> float:
        """
        选择动作并返回实际的动作值

        Args:
            state: 当前状态
            training: 是否处于训练模式

        Returns:
            实际的动作值
        """
        action_idx = self.select_action(state, training)
        return self.get_action_value(action_idx)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        使用 ε-贪心策略选择动作索引

        Args:
            state: 当前状态
            training: 是否处于训练模式

        Returns:
            动作索引
        """
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if np.random.random() > self.epsilon:  # 利用
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                    0
                )
                q_values = self.policy_net(state_t)
                action_idx = torch.argmax(q_values[0][: self.action_size]).item()
                return int(action_idx)
        else:  # 探索
            return int(np.random.randint(0, self.action_size))

    def store_transition(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        """
        存储转移到重放缓冲区

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
        """
        self.replay_buffer.push(state=state, action=action, reward=reward, next_state=next_state)

    def learn(self) -> float:
        """
        从重放缓冲区中采样并更新策略网络

        Returns:
            损失值
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        if self.learn_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_counter += 1

        # 从重放缓冲区采样小批量
        batch = self.replay_buffer.sample(self.batch_size)
        b_s = batch["state"]
        b_a = batch["action"]
        b_r = batch["reward"]
        b_s_next = batch["next_state"]

        # 转换为张量
        b_s = torch.as_tensor(b_s, dtype=torch.float32, device=self.device)
        b_a = torch.as_tensor(b_a, dtype=torch.long, device=self.device)
        b_r = torch.as_tensor(b_r, dtype=torch.float32, device=self.device)
        b_s_next = torch.as_tensor(b_s_next, dtype=torch.float32, device=self.device)

        # 计算 Q 学习目标
        q_pred = self.policy_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_next).detach()
        target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        # 优化
        loss = self.loss_func(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, filepath: str, episode: int = 0, **kwargs) -> None:
        """
        保存智能体检查点

        Args:
            filepath: 保存路径
            episode: 当前轮数
            **kwargs: 其他元数据
        """
        checkpoint = {
            "model_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode,
            "epsilon": float(self.epsilon),
            "replay_buffer_size": len(self.replay_buffer),
            "learn_counter": self.learn_counter,
            "model_config": {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "hidden_size": self.hidden_size,
                "gamma": self.gamma,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "memory_capacity": self.memory_capacity,
                "batch_size": self.batch_size,
                "target_update": self.target_update,
            },
            "action_space_values": (
                self.action_space.tolist()
                if isinstance(self.action_space, np.ndarray)
                else list(self.action_space)
            ),
        }
        checkpoint.update(kwargs)
        torch.save(checkpoint, filepath)

    @staticmethod
    def load_checkpoint(
        filepath: str,
        device: torch.device,
        action_space: Optional[np.ndarray] = None,
        policy_net_class: Optional[type] = None,
        policy_net_kwargs: Optional[dict] = None,
    ) -> Tuple["DQNAgent", dict]:
        """
        从检查点加载智能体

        Args:
            filepath: 检查点文件路径
            device: PyTorch 设备（必需）
            action_space: 可选的动作空间数组
            policy_net_class: 自定义网络类（可选）
            policy_net_kwargs: 网络参数字典（可选）

        Returns:
            (智能体实例, 检查点字典)
        """
        if device is None:
            raise ValueError("'device' is required")

        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint["model_config"]

        if action_space is None:
            if "action_space_values" in checkpoint:
                action_space = np.array(checkpoint["action_space_values"])
            else:
                raise ValueError("Action space not found in checkpoint and not provided")

        agent = DQNAgent(
            state_size=config["state_size"],
            action_size=config["action_size"],
            action_space=action_space,
            device=device,
            gamma=config["gamma"],
            epsilon_min=config["epsilon_min"],
            epsilon_decay=config["epsilon_decay"],
            memory_capacity=config["memory_capacity"],
            batch_size=config["batch_size"],
            target_update=config["target_update"],
            hidden_size=config["hidden_size"],
            policy_net_class=policy_net_class,
            policy_net_kwargs=policy_net_kwargs,
        )

        agent.policy_net.load_state_dict(checkpoint["model_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
        agent.learn_counter = checkpoint.get("learn_counter", 0)

        return agent, checkpoint

    def get_action_value(self, action_idx: int) -> float:
        """
        从动作索引获取实际动作值

        Args:
            action_idx: 动作索引

        Returns:
            实际动作值
        """
        value = self.action_space[action_idx]
        if isinstance(value, np.ndarray):
            return float(value.item())
        return float(value)
