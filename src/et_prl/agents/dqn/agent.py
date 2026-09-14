from __future__ import annotations

import random

import numpy as np
import torch
from et_prl.agents.dqn import DQNAgent as ToolkitDQNAgent
from et_prl.models import QNetwork
from et_prl.utils import create_optimizer


class DQNAgent(ToolkitDQNAgent):
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
        hidden_sizes = hidden_sizes or [128]
        super().__init__(
            state_size=state_size,
            action_size=len(action_space),
            action_space=action_space,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            memory_capacity=memory_capacity,
            batch_size=batch_size,
            target_update=target_update_freq,
            hidden_size=hidden_sizes[0],
            policy_net_class=QNetwork,
            policy_net_kwargs={"hidden_sizes": hidden_sizes},
        )
        self.target_update_freq = target_update_freq
        self.optimizer = create_optimizer(self.policy_net, "adam", learning_rate)
        self.loss_func = torch.nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def get_action_value(self, action_idx: int) -> float:
        return float(self.action_space[action_idx])

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ) -> None:
        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
        )

    def learn(self) -> float:
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

    def save_checkpoint(self, filepath: str, episode: int = 0, **kwargs) -> None:
        state = {
            "episode": episode,
            "best_reward": kwargs.get("best_reward"),
            "model_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        state.update(kwargs)
        torch.save(state, filepath)

