from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ml_toolkit.rl import SequenceEnv
from ml_toolkit.utils import CheckpointManager, Logger, MetricsRecorder

from config.dqn_config import DQNConfig
from src.dqn.agent import DQNAgent
from src.dqn.rewards import RewardCalculator
from src.dqn.trainer import DQNTrainer


def _resolve_train_experiment_dir(train_experiment_dir: Path | None) -> Path:
    if train_experiment_dir is not None:
        resolved = Path(train_experiment_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"指定的训练实验目录不存在: {resolved}")
        return resolved

    latest_dir = CheckpointManager.find_latest_experiment(
        experiment_name=DQNConfig.EXPERIMENT_NAME,
        mode="train",
        log_root_dir=DQNConfig.LOG_ROOT_DIR,
    )
    if latest_dir is None:
        raise FileNotFoundError("未找到 DQN 训练实验目录，请先执行训练")
    return latest_dir


def _create_agent(config: type[DQNConfig], action_space: np.ndarray, device: torch.device) -> DQNAgent:
    return DQNAgent(
        state_size=config.STATE_SIZE,
        action_space=action_space,
        hidden_sizes=config.HIDDEN_SIZES,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        epsilon_start=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay=config.EPSILON_DECAY,
        memory_capacity=config.MEMORY_CAPACITY,
        batch_size=config.BATCH_SIZE,
        target_update_freq=config.TARGET_UPDATE_FREQ,
        device=device,
    )


def _evaluate_episode(agent: DQNAgent, env: SequenceEnv) -> tuple[dict, pd.DataFrame]:
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    energy_scores: list[float] = []
    comfort_scores: list[float] = []
    records: list[dict] = []

    while True:
        action_idx = agent.select_action(state, training=False)
        action_value = agent.get_action_value(action_idx)
        next_state, reward, terminated, _, info = env.step(action_value)

        steps += 1
        reward_value = float(reward)
        energy_score = float(info.get("energy_score", 0.0))
        comfort_score = float(info.get("comfort_score", 0.0))

        total_reward += reward_value
        energy_scores.append(energy_score)
        comfort_scores.append(comfort_score)

        records.append(
            {
                "step": steps,
                "action_idx": int(action_idx),
                "action_value": float(action_value),
                "reward": reward_value,
                "energy_score": energy_score,
                "comfort_score": comfort_score,
            }
        )

        state = next_state
        if terminated:
            break

    avg_reward_per_step = total_reward / steps if steps > 0 else 0.0
    summary = {
        "steps": steps,
        "total_reward": float(total_reward),
        "avg_reward_per_step": float(avg_reward_per_step),
        "avg_energy_score": float(np.mean(energy_scores)) if energy_scores else 0.0,
        "avg_comfort_score": float(np.mean(comfort_scores)) if comfort_scores else 0.0,
    }

    return summary, pd.DataFrame(records)


def eval_dqn(train_experiment_dir: Path | None = None) -> None:
    config = DQNConfig
    eval_experiment_dir = config.get_eval_experiment_dir()

    logger = Logger(eval_experiment_dir, log_filename=config.EVALUATION_LOG_FILENAME)
    metrics_recorder = MetricsRecorder(eval_experiment_dir, metrics_filename=config.EVALUATION_METRICS_FILENAME)

    resolved_train_dir = _resolve_train_experiment_dir(train_experiment_dir)
    logger.info(f"使用训练实验目录: {resolved_train_dir}")

    train_config_path = resolved_train_dir / config.CONFIG_FILENAME
    eval_config_path = eval_experiment_dir / config.CONFIG_FILENAME
    if train_config_path.exists():
        shutil.copy2(train_config_path, eval_config_path)
        logger.info(f"已复制训练配置: {eval_config_path}")
    else:
        logger.warning(f"训练配置文件不存在，跳过复制: {train_config_path}")

    test_data = DQNTrainer._load_data(config.get_test_data_path(), config.STATE_COLUMNS)
    if test_data.empty:
        raise ValueError(f"测试数据为空: {config.get_test_data_path()}")

    action_space = DQNTrainer._load_action_space(config.ACTION_SPACE_PATH)
    device = torch.device(config.DEVICE)

    agent = _create_agent(config, action_space, device)
    checkpoint = CheckpointManager.load_best_model(
        experiment_dir=resolved_train_dir,
        model=agent.policy_net,
        best_filename=config.BEST_MODEL_FILENAME,
        checkpoint_dir_name=config.CHECKPOINT_DIR_NAME,
        map_location=device,
    )
    agent.policy_net.eval()

    reward_calc = RewardCalculator(
        data=test_data,
        action_space=action_space,
        weight_efficiency=config.REWARD_WEIGHT_EFFICIENCY,
        weight_comfort=config.REWARD_WEIGHT_COMFORT,
        target_supply_temp=config.TARGET_SUPPLY_TEMP,
        coeff_path=config.COEFF_DATE_PATH,
        chiller_capacity=config.CHILLER_CAPACITY,
        chiller_ref_power=config.CHILLER_REF_POWER,
        supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
        comfort_sigma=config.COMFORT_SIGMA,
        chiller_high_threshold=config.CHILLER_HIGH_THRESHOLD,
        chiller_medium_threshold=config.CHILLER_MEDIUM_THRESHOLD,
        chiller_low_threshold=config.CHILLER_LOW_THRESHOLD,
        f_nominal=config.CHILLER_F_NOMINAL,
        f_cw=config.CHILLER_F_CW,
        f_tower=config.CHILLER_F_TOWER,
        f_chw=config.CHILLER_F_CHW,
        c_p=config.CHILLER_CP,
        density_water=config.CHILLER_WATER_DENSITY,
    )
    test_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)

    summary, step_results = _evaluate_episode(agent, test_env)

    action_distribution = (
        step_results["action_idx"].value_counts().sort_index().to_dict() if not step_results.empty else {}
    )
    action_distribution = {str(k): int(v) for k, v in action_distribution.items()}

    output_payload = {
        "train_experiment_dir": str(resolved_train_dir),
        "best_epoch_from_train": int(checkpoint.get("epoch", -1)) + 1,
        "best_metrics_from_train": checkpoint.get("metrics", {}),
        "test_summary": summary,
        "action_distribution": action_distribution,
    }
    metrics_recorder.save_metrics(output_payload)

    step_results_path = eval_experiment_dir / "test_step_results.csv"
    step_results.to_csv(step_results_path, index=False)

    logger.info(f"评估完成: steps={summary['steps']}, total_reward={summary['total_reward']:.4f}")
    logger.info(
        "关键指标: "
        f"avg_reward_per_step={summary['avg_reward_per_step']:.4f}, "
        f"avg_energy_score={summary['avg_energy_score']:.4f}, "
        f"avg_comfort_score={summary['avg_comfort_score']:.4f}"
    )
    logger.info(f"评估指标已保存: {eval_experiment_dir / config.EVALUATION_METRICS_FILENAME}")
    logger.info(f"逐步结果已保存: {step_results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN 模型评估脚本")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="指定训练实验目录路径；不指定则自动加载最新训练实验",
    )
    args = parser.parse_args()

    eval_dqn(Path(args.experiment_dir) if args.experiment_dir else None)


if __name__ == "__main__":
    main()
