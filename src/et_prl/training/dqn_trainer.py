from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from et_prl.agents.dqn import DQNAgent
from et_prl.agents.dqn.rewards import RewardCalculator as _RewardCalculator
from et_prl.config.dqn import DQNConfig
from et_prl.data import load_action_space, load_state_data
from et_prl.environments import SequenceEnv
from et_prl.utils import CheckpointManager, ExperimentContext, Logger, MetricsRecorder


class DQNTrainer:
    def __init__(
        self,
        config: DQNConfig,
        experiment_dir: Path,
        context: ExperimentContext | None = None,
    ) -> None:
        self.config = config
        self.experiment_dir = experiment_dir

        # 复用实验上下文的 logger/metrics_recorder，避免同一 run.log 双文件句柄与重复 TB writer；
        # 未注入 context 时（如超参优化 trial）回退为自建。
        if context is not None:
            self.logger = context.logger
            self.metrics_recorder = context.metrics_recorder
            self.checkpoint_manager = context.checkpoint_manager or CheckpointManager(
                experiment_dir, checkpoint_dir_name=config.CHECKPOINT_DIR_NAME
            )
        else:
            self.logger = Logger(experiment_dir)
            self.metrics_recorder = MetricsRecorder(experiment_dir)
            self.checkpoint_manager = CheckpointManager(
                experiment_dir, checkpoint_dir_name=config.CHECKPOINT_DIR_NAME
            )

    def train(self) -> None:
        self.config.validate()
        self.logger.info("配置校验完成")
        self.logger.info(
            "训练随机性参数: "
            f"RANDOM_STATE={getattr(self.config, 'RANDOM_STATE', None)}, "
            f"CUDNN_DETERMINISTIC={getattr(self.config, 'CUDNN_DETERMINISTIC', None)}"
        )

        train_data = load_state_data(self.config.get_train_data_path(), self.config.STATE_COLUMNS)
        val_data = load_state_data(self.config.get_val_data_path(), self.config.STATE_COLUMNS)
        action_space = load_action_space(self.config.ACTION_SPACE_PATH)

        device = torch.device(self.config.DEVICE)
        agent = DQNAgent.from_config(config=self.config, action_space=action_space, device=device)

        reward_calc = _RewardCalculator.from_config(
            config=self.config, data=train_data, action_space=action_space
        )
        train_env = SequenceEnv(train_data, self.config.STATE_COLUMNS, reward_calc)
        val_reward_calc = _RewardCalculator.from_config(
            config=self.config, data=val_data, action_space=action_space
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
                val_info = self._val_episode(agent, val_env)
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

    def _val_episode(self, agent: DQNAgent, env: SequenceEnv) -> dict:
        # select_action(training=False) 本身即纯贪心（不读取 epsilon），无需临时改写 agent.epsilon
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            steps += 1
            action_idx = agent.select_action(state, training=False)
            action_value = agent.get_action_value(action_idx)
            next_state, reward, terminated, _, _ = env.step(action_value)
            total_reward += float(reward)
            state = next_state
            if terminated:
                break
        return {"total_reward": total_reward, "steps": steps}

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
        output_path = self.config.get_run_results_dir() / self.config.TRAINING_HISTORY_FILENAME
        self.logger.info(f"训练历史已保存: {output_path}")
