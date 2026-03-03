from __future__ import annotations

from pathlib import Path

from ml_toolkit.rl import SequenceEnv
from ml_toolkit.utils import create_experiment_context

from config.dqn_config import DQNConfig
from src.control_evaluation.common import (
    build_eval_components,
    copy_train_config,
    get_paper_symbol_field_mapping,
    resolve_train_experiment_dir,
)
from src.control_evaluation.strategies import evaluate_fixed_interval


def eval_dqn(train_experiment_dir: Path | None = None, fixed_interval: int = 4) -> None:
    config = DQNConfig
    eval_experiment_dir = config.get_eval_experiment_dir()

    context = create_experiment_context(
        experiment_dir=eval_experiment_dir,
        config=config,
        save_config=False,
        log_filename=config.EVALUATION_LOG_FILENAME,
        metrics_filename=config.EVALUATION_METRICS_FILENAME,
        config_filename=config.CONFIG_FILENAME,
    )
    logger = context.logger
    metrics_recorder = context.metrics_recorder

    if fixed_interval <= 0:
        raise ValueError(f"fixed_interval 必须大于 0，当前: {fixed_interval}")

    resolved_train_dir = resolve_train_experiment_dir(train_experiment_dir)
    logger.info(f"使用训练实验目录: {resolved_train_dir}")

    copy_train_config(
        resolved_train_dir=resolved_train_dir,
        eval_experiment_dir=eval_experiment_dir,
        config=config,
        logger=logger,
    )

    test_data, action_space, agent, checkpoint, reward_calc = build_eval_components(
        config=config,
        resolved_train_dir=resolved_train_dir,
    )
    test_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)

    summary, step_results = evaluate_fixed_interval(
        agent=agent,
        env=test_env,
        action_space=action_space,
        fixed_interval=fixed_interval,
        supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
    )

    action_distribution = (
        step_results["action_idx"].value_counts().sort_index().to_dict() if not step_results.empty else {}
    )
    action_distribution = {str(k): int(v) for k, v in action_distribution.items()}

    output_payload = {
        "train_experiment_dir": str(resolved_train_dir),
        "best_epoch_from_train": int(checkpoint.get("epoch", -1)) + 1,
        "best_metrics_from_train": checkpoint.get("metrics", {}),
        "mode": "fixed_interval",
        "fixed_interval": int(fixed_interval),
        "paper_symbol_mapping": get_paper_symbol_field_mapping(),
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
        f"avg_comfort_score={summary['avg_comfort_score']:.4f}, "
        f"action_count={summary['action_count']}, "
        f"action_frequency={summary['action_frequency']:.4f}"
    )
    logger.info(f"评估指标已保存: {eval_experiment_dir / config.EVALUATION_METRICS_FILENAME}")
    logger.info(f"逐步结果已保存: {step_results_path}")
