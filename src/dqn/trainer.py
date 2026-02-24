from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ml_toolkit.rl import SequenceEnv
from ml_toolkit.utils import CheckpointManager, Logger, MetricsRecorder

from config.dqn_config import DQNConfig
from src.dqn.agent import DQNAgent
from src.dqn.rewards import RewardCalculator


class DQNTrainer:
    def __init__(self, config: type[DQNConfig], experiment_dir: Path) -> None:
        self.config = config
        self.experiment_dir = experiment_dir

        self.logger = Logger(experiment_dir)
        self.metrics_recorder = MetricsRecorder(experiment_dir)
        self.checkpoint_manager = CheckpointManager(experiment_dir, checkpoint_dir_name=config.CHECKPOINT_DIR_NAME)

    def train(self) -> None:
        self.config.validate()
        self.logger.info("配置校验完成")

        train_data = self._load_data(self.config.get_train_data_path(), self.config.STATE_COLUMNS)
        val_data = self._load_data(self.config.get_val_data_path(), self.config.STATE_COLUMNS)
        action_space = self._load_action_space(self.config.ACTION_SPACE_PATH)

        device = torch.device(self.config.DEVICE)
        agent = DQNAgent(
            state_size=self.config.STATE_SIZE,
            action_space=action_space,
            hidden_sizes=self.config.HIDDEN_SIZES,
            learning_rate=self.config.LEARNING_RATE,
            gamma=self.config.GAMMA,
            epsilon_start=self.config.EPSILON_START,
            epsilon_min=self.config.EPSILON_MIN,
            epsilon_decay=self.config.EPSILON_DECAY,
            memory_capacity=self.config.MEMORY_CAPACITY,
            batch_size=self.config.BATCH_SIZE,
            target_update_freq=self.config.TARGET_UPDATE_FREQ,
            device=device,
        )

        reward_calc = RewardCalculator(
            train_data,
            action_space,
            self.config.REWARD_WEIGHT_EFFICIENCY,
            self.config.REWARD_WEIGHT_COMFORT,
            self.config.TARGET_SUPPLY_TEMP,
        )
        train_env = SequenceEnv(train_data, self.config.STATE_COLUMNS, reward_calc)
        val_reward_calc = RewardCalculator(
            val_data,
            action_space,
            self.config.REWARD_WEIGHT_EFFICIENCY,
            self.config.REWARD_WEIGHT_COMFORT,
            self.config.TARGET_SUPPLY_TEMP,
        )
        val_env = SequenceEnv(val_data, self.config.STATE_COLUMNS, val_reward_calc)

        self.logger.info("开始训练")
        best_val_reward = float("-inf")
        best_episode = 0
        no_improvement = 0
        training_history: list[dict] = []

        for episode in range(self.config.NUM_EPISODES):
            episode_info = self._train_episode(agent, train_env)
            train_record = {
                "episode": episode + 1,
                "train_reward": episode_info["total_reward"],
                "train_loss": episode_info["avg_loss"],
                "train_steps": episode_info["steps"],
                "train_avg_energy_score": episode_info["avg_energy_score"],
                "train_avg_comfort_score": episode_info["avg_comfort_score"],
                "epsilon": agent.epsilon,
            }

            self.logger.info(
                f"Episode {episode + 1}/{self.config.NUM_EPISODES} | "
                f"reward={episode_info['total_reward']:.4f} | "
                f"energy={episode_info['avg_energy_score']:.4f} | "
                f"comfort={episode_info['avg_comfort_score']:.4f}"
            )

            if self._should_validate(episode):
                val_info = self._eval_episode(agent, val_env)
                val_reward = val_info["total_reward"]
                train_record["val_reward"] = val_reward
                train_record["val_steps"] = val_info["steps"]
                self.logger.info(f"验证奖励: {val_reward:.4f}")

                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    best_episode = episode
                    no_improvement = 0
                    self.checkpoint_manager.save_checkpoint(
                        model=agent.policy_net,
                        optimizer=agent.optimizer,
                        epoch=episode,
                        metrics={
                            "val_reward": val_reward,
                            "epsilon": float(agent.epsilon),
                        },
                        config=self.config.to_dict(),
                        filename="checkpoint.pth",
                        is_best=True,
                        best_filename=self.config.BEST_MODEL_FILENAME,
                    )
                    self.logger.info("已保存最优模型")
                else:
                    no_improvement += 1
                    if self._should_stop_early(no_improvement, episode):
                        self.logger.warning("触发早停，训练结束")
                        training_history.append(train_record)
                        break
            else:
                train_record["val_reward"] = np.nan
                train_record["val_steps"] = np.nan

            training_history.append(train_record)

        self.checkpoint_manager.save_checkpoint(
            model=agent.policy_net,
            optimizer=agent.optimizer,
            epoch=len(training_history) - 1,
            metrics={
                "best_val_reward": best_val_reward,
                "epsilon": float(agent.epsilon),
            },
            config=self.config.to_dict(),
            filename=self.config.FINAL_MODEL_FILENAME,
            is_best=False,
        )
        self.logger.info("已保存最终模型")

        self._save_training_history(training_history)
        self.metrics_recorder.save_metrics(
            {
                "summary": {
                    "best_episode": best_episode + 1,
                    "best_val_reward": best_val_reward,
                    "total_episodes": len(training_history),
                }
            }
        )
        self.logger.info("训练完成")

    def _train_episode(self, agent: DQNAgent, env: SequenceEnv) -> dict:
        state, _ = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        energy_scores: list[float] = []
        comfort_scores: list[float] = []
        steps = 0

        while True:
            steps += 1
            action_idx = agent.select_action(state, training=True)
            action_value = agent.get_action_value(action_idx)
            next_state, reward, terminated, _, info = env.step(action_value)

            agent.store_transition(state, action_idx, reward, next_state)
            loss = agent.learn()
            total_loss += loss

            total_reward += float(reward)
            energy_scores.append(float(info.get("energy_score", 0.0)))
            comfort_scores.append(float(info.get("comfort_score", 0.0)))

            state = next_state
            if terminated:
                break

        avg_energy = float(np.mean(energy_scores)) if energy_scores else 0.0
        avg_comfort = float(np.mean(comfort_scores)) if comfort_scores else 0.0
        avg_loss = total_loss / steps if steps > 0 else 0.0

        return {
            "total_reward": total_reward,
            "total_loss": total_loss,
            "avg_loss": float(avg_loss),
            "steps": steps,
            "avg_energy_score": avg_energy,
            "avg_comfort_score": avg_comfort,
        }

    def _eval_episode(self, agent: DQNAgent, env: SequenceEnv) -> dict:
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        epsilon_backup = agent.epsilon
        agent.epsilon = 0.0

        while True:
            steps += 1
            action_idx = agent.select_action(state, training=False)
            action_value = agent.get_action_value(action_idx)
            next_state, reward, terminated, _, _ = env.step(action_value)
            total_reward += float(reward)
            state = next_state
            if terminated:
                break

        agent.epsilon = epsilon_backup
        return {"total_reward": total_reward, "steps": steps}

    @staticmethod
    def _load_data(path: Path, required_columns: list[str]) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        data = pd.read_csv(path)
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"数据缺少必要列: {missing}")
        return data

    @staticmethod
    def _load_action_space(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"动作空间文件不存在: {path}")
        action_space = np.load(path, allow_pickle=True)
        action_space = np.asarray(action_space, dtype=np.float32).squeeze()

        if action_space.ndim == 0:
            action_space = action_space.reshape(1)

        if action_space.ndim != 1:
            raise ValueError(f"动作空间维度必须为 1，当前维度: {action_space.ndim}，形状: {action_space.shape}")

        if action_space.size == 0:
            raise ValueError("动作空间不能为空")

        return action_space

    def _should_validate(self, episode: int) -> bool:
        if self.config.VAL_INTERVAL <= 0:
            return False
        return episode % self.config.VAL_INTERVAL == 0 or episode == self.config.NUM_EPISODES - 1

    def _should_stop_early(self, no_improvement: int, episode: int) -> bool:
        if self.config.EARLY_STOPPING_PATIENCE <= 0:
            return False
        if episode <= self.config.NUM_EPISODES // 2:
            return False
        return no_improvement >= self.config.EARLY_STOPPING_PATIENCE

    def _save_training_history(self, history: list[dict]) -> None:
        if not history:
            return
        train_metrics = [
            {
                "reward": record["train_reward"],
                "steps": record["train_steps"],
                "avg_energy_score": record["train_avg_energy_score"],
                "avg_comfort_score": record["train_avg_comfort_score"],
                "epsilon": record["epsilon"],
            }
            for record in history
        ]
        val_metrics = [
            {
                "reward": record.get("val_reward", np.nan),
                "steps": record.get("val_steps", np.nan),
            }
            for record in history
        ]

        history_dict = {
            "train_loss": [record.get("train_loss", np.nan) for record in history],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        self.metrics_recorder.save_training_history(history_dict)
        output_path = self.experiment_dir / self.config.TRAINING_HISTORY_FILENAME
        self.logger.info(f"训练历史已保存: {output_path}")





