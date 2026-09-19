from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from et_prl.agents.dqn import DQNAgent
from et_prl.agents.dqn.rewards import RewardCalculator
from et_prl.config.dqn import DQNConfig
from et_prl.config.loader import load_config
from et_prl.data import load_action_space, load_state_data
from et_prl.environments import SequenceEnv
from et_prl.training.dqn_trainer import DQNTrainer
from et_prl.utils import BayesianOptimizer, HyperparameterSpace, Logger, configure_reproducibility


def _parse_hidden_layout(layout: str) -> list[int]:
    return [int(part) for part in str(layout).split(",") if part]


def _create_search_space() -> HyperparameterSpace:
    space = HyperparameterSpace()
    (
        space.add_float("learning_rate", 5e-5, 5e-3, log=True)
        .add_float("gamma", 0.90, 0.995)
        .add_float("epsilon_start", 0.6, 1.0)
        .add_float("epsilon_min", 0.001, 0.1, log=True)
        .add_int("batch_size", 32, 256)
        .add_float("epsilon_decay", 0.990, 0.9995)
        .add_int("target_update_freq", 50, 300)
        .add_categorical(
            "hidden_layout",
            [
                "128",
                "192",
                "256",
                "320",
                "192,96",
                "256,128",
                "320,160",
                "256,128,64",
                "320,160,80",
            ],
        )
        .add_categorical("memory_capacity", [20000, 50000, 80000])
        .add_float("weight_efficiency", 0.30, 0.80)
        .add_float("comfort_sigma", 1.0, 3.0)
    )
    return space


def _build_trial_config(base_config, params: dict, max_episodes: int):
    """基于 base 配置实例生成 trial 配置（frozen dataclass 用 replace）。"""
    weight_efficiency = float(params["weight_efficiency"])
    updates = {
        "LEARNING_RATE": float(params["learning_rate"]),
        "GAMMA": float(params["gamma"]),
        "EPSILON_START": float(params["epsilon_start"]),
        "EPSILON_MIN": float(params["epsilon_min"]),
        "BATCH_SIZE": int(params["batch_size"]),
        "EPSILON_DECAY": float(params["epsilon_decay"]),
        "TARGET_UPDATE_FREQ": int(params["target_update_freq"]),
        "MEMORY_CAPACITY": int(params["memory_capacity"]),
        "HIDDEN_SIZES": _parse_hidden_layout(params["hidden_layout"]),
        "COMFORT_SIGMA": float(params["comfort_sigma"]),
        "REWARD_WEIGHT_EFFICIENCY": weight_efficiency,
        "REWARD_WEIGHT_COMFORT": 1.0 - weight_efficiency,
        "NUM_EPISODES": int(max_episodes),
        "VAL_INTERVAL": max(1, min(base_config.VAL_INTERVAL, max_episodes)),
        "EARLY_STOPPING_PATIENCE": max(
            2, min(base_config.EARLY_STOPPING_PATIENCE, max_episodes // 3 or 2)
        ),
    }
    return replace(base_config, **updates)


def _create_objective(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    action_space,
    base_config: DQNConfig,
    max_episodes: int,
    output_dir: Path,
    logger: Logger,
):
    active_trials = {"count": 0}
    active_lock = threading.Lock()

    def objective(trial, params: dict) -> float:
        trial_start = time.perf_counter()
        worker_name = threading.current_thread().name
        with active_lock:
            active_trials["count"] += 1
            current_active = active_trials["count"]

        logger.info(
            f"Trial {trial.number} 开始 | worker={worker_name} | "
            f"active_trials={current_active} | "
            f"lr={params['learning_rate']:.2e}, bs={int(params['batch_size'])}, "
            f"hidden={params['hidden_layout']}"
        )

        try:
            trial_config = _build_trial_config(base_config, params, max_episodes)
            trial_config.validate()

            trial_dir = output_dir / "trials" / f"trial_{trial.number:04d}"
            trainer = DQNTrainer(trial_config, trial_dir)
            device = torch.device(trial_config.DEVICE)

            agent = DQNAgent.from_config(
                config=trial_config, action_space=action_space, device=device
            )

            train_reward_calc = RewardCalculator.from_config(
                config=trial_config, data=train_data, action_space=action_space
            )
            val_reward_calc = RewardCalculator.from_config(
                config=trial_config, data=val_data, action_space=action_space
            )
            train_env = SequenceEnv(train_data, trial_config.STATE_COLUMNS, train_reward_calc)
            val_env = SequenceEnv(val_data, trial_config.STATE_COLUMNS, val_reward_calc)

            best_val_reward = float("-inf")
            no_improvement = 0

            for episode in range(trial_config.NUM_EPISODES):
                trainer._train_episode(agent, train_env)

                if trainer._should_validate(episode):
                    val_info = trainer._val_episode(agent, val_env)
                    val_reward = float(val_info["total_reward"])

                    if val_reward > best_val_reward:
                        best_val_reward = val_reward
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if trainer._should_stop_early(no_improvement, episode):
                            break

            if best_val_reward == float("-inf"):
                best_val_reward = float("-1e9")

            trial.set_user_attr("best_val_reward", best_val_reward)
            elapsed_sec = time.perf_counter() - trial_start
            logger.info(
                f"Trial {trial.number} 完成: best_val_reward={best_val_reward:.6f}, "
                f"lr={trial_config.LEARNING_RATE:.2e}, bs={trial_config.BATCH_SIZE}, "
                f"hidden={trial_config.HIDDEN_SIZES}, elapsed={elapsed_sec:.1f}s"
            )

            return -best_val_reward

        except Exception as exc:
            elapsed_sec = time.perf_counter() - trial_start
            logger.exception(f"Trial {trial.number} 失败: {exc}")
            logger.info(
                f"Trial {trial.number} 结束(失败) | worker={worker_name} | "
                f"elapsed={elapsed_sec:.1f}s"
            )
            return 1e9

        finally:
            with active_lock:
                active_trials["count"] -= 1
                current_active = active_trials["count"]
            logger.info(
                f"Trial {trial.number} 释放 worker={worker_name} | active_trials={current_active}"
            )

    return objective


def optimize_dqn_hyperparameters(
    n_trials: int = 30, max_episodes: int = 30, n_jobs: int = 1
) -> dict:
    base_config = load_config("dqn")
    output_dir = base_config.get_optimization_dir() / base_config.TIMESTAMP
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(output_dir, log_filename="run.log")
    reproducibility = configure_reproducibility(
        seed=int(base_config.RANDOM_STATE),
        deterministic_cudnn=bool(base_config.CUDNN_DETERMINISTIC),
    )
    logger.info(
        "随机性配置(优化全局): "
        f"seed={reproducibility['seed']}, "
        f"deterministic_cudnn={reproducibility['deterministic_cudnn']}, "
        f"cudnn_benchmark={reproducibility['cudnn_benchmark']}, "
        f"cuda_available={reproducibility['cuda_available']}"
    )
    logger.info("开始 DQN 贝叶斯超参优化")
    logger.info(f"试验次数: {n_trials}, 每个 trial 最大轮数: {max_episodes}, 并行任务数: {n_jobs}")

    train_data = load_state_data(base_config.get_train_data_path(), base_config.STATE_COLUMNS)
    val_data = load_state_data(base_config.get_val_data_path(), base_config.STATE_COLUMNS)
    action_space = load_action_space(base_config.ACTION_SPACE_PATH)

    space = _create_search_space()
    optimizer = BayesianOptimizer(
        space=space,
        output_dir=output_dir,
        results_dir=output_dir / "results",
        sampler="tpe",
        seed=int(base_config.RANDOM_STATE),
    )

    objective = _create_objective(
        train_data=train_data,
        val_data=val_data,
        action_space=action_space,
        base_config=base_config,
        max_episodes=max_episodes,
        output_dir=output_dir,
        logger=logger,
    )

    result = optimizer.optimize(objective_fn=objective, n_trials=n_trials, n_jobs=n_jobs)

    best_val_reward = -float(result["best_value"])
    best_params = result["best_params"]
    best_config = {
        "LEARNING_RATE": float(best_params["learning_rate"]),
        "GAMMA": float(best_params["gamma"]),
        "EPSILON_START": float(best_params["epsilon_start"]),
        "EPSILON_MIN": float(best_params["epsilon_min"]),
        "BATCH_SIZE": int(best_params["batch_size"]),
        "EPSILON_DECAY": float(best_params["epsilon_decay"]),
        "TARGET_UPDATE_FREQ": int(best_params["target_update_freq"]),
        "MEMORY_CAPACITY": int(best_params["memory_capacity"]),
        "HIDDEN_SIZES": _parse_hidden_layout(best_params["hidden_layout"]),
        "COMFORT_SIGMA": float(best_params["comfort_sigma"]),
        "REWARD_WEIGHT_EFFICIENCY": float(best_params["weight_efficiency"]),
        "REWARD_WEIGHT_COMFORT": float(1.0 - best_params["weight_efficiency"]),
    }

    summary = {
        "n_trials": int(result["n_trials"]),
        "best_val_reward": best_val_reward,
        "best_params": best_params,
        "best_config_overrides": best_config,
    }

    summary_path = output_dir / "results" / "best_config.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    if optimizer.study is not None:
        trial_records = []
        for trial in optimizer.study.trials:
            trial_records.append(
                {
                    "trial_number": trial.number,
                    "state": trial.state.name,
                    "objective_value": trial.value,
                    "best_val_reward": -trial.value if trial.value is not None else None,
                    **trial.params,
                }
            )
        pd.DataFrame(trial_records).to_csv(output_dir / "results" / "trials.csv", index=False)

    logger.info(f"优化完成，最优验证奖励: {best_val_reward:.6f}")
    logger.info(f"最优配置已保存: {summary_path}")

    return summary
