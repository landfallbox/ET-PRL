from __future__ import annotations

from pathlib import Path

from et_prl.environments import SequenceEnv
from et_prl.utils import create_experiment_context

from et_prl.config.dqn import DQNConfig
from et_prl.config.event_driven_dqn import EventDrivenDQNConfig
from et_prl.evaluation.control.common import (
    build_test_components,
    copy_train_config,
    create_streaming_gate,
    get_paper_symbol_field_mapping,
    resolve_train_experiment_dir,
)
from et_prl.evaluation.control.strategies import test_event_driven as test_event_driven_strategy
from et_prl.config.loader import load_config


def test_event_driven(
    train_experiment_dir: Path | None = None,
    gate_state_path: Path | None = None,
) -> None:
    config = load_config("event_driven")
    test_experiment_dir = config.get_eval_experiment_dir()

    context = create_experiment_context(
        experiment_dir=test_experiment_dir,
        config=config,
        save_config=False,
        log_filename=config.RUN_LOG_FILENAME,
        config_filename=config.CONFIG_FILENAME,
        results_dir=config.get_run_results_dir(mode="eval"),
        tb_dir=config.get_run_tb_dir(mode="eval"),
    )
    logger = context.logger
    metrics_recorder = context.metrics_recorder

    resolved_train_dir = resolve_train_experiment_dir(train_experiment_dir)
    logger.info(f"使用训练实验目录: {resolved_train_dir}")

    copy_train_config(
        resolved_train_dir=resolved_train_dir,
        test_experiment_dir=test_experiment_dir,
        config=config,
        logger=logger,
    )

    test_data, action_space, agent, checkpoint, reward_calc = build_test_components(
        config=config,
        resolved_train_dir=resolved_train_dir,
    )

    event_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
    event_gate = create_streaming_gate(config=config, test_data=test_data, logger=logger, gate_state_path=gate_state_path)
    summary, step_results = test_event_driven_strategy(
        agent=agent,
        env=event_env,
        data=test_data,
        action_space=action_space,
        gate=event_gate,
        feature_columns=config.FEATURE_COLUMNS,
        supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
    )

    output_payload = {
        "train_experiment_dir": str(resolved_train_dir),
        "best_epoch_from_train": int(checkpoint.get("epoch", -1)) + 1,
        "best_metrics_from_train": checkpoint.get("metrics", {}),
        "mode": "event_driven",
        "gate_state_path": str(gate_state_path) if gate_state_path is not None else None,
        "paper_symbol_mapping": get_paper_symbol_field_mapping(),
        "event_driven_summary": summary,
    }
    metrics_recorder.save_metrics(output_payload)

    step_results_path = config.get_run_results_dir(mode="eval") / "event_driven_step_results.csv"
    step_results.to_csv(step_results_path, index=False)

    logger.info(
        "事件驱动评估完成: "
        f"total_reward={summary['total_reward']:.4f}, "
        f"action_count={summary['action_count']}, "
        f"event_trigger_rate={summary['event_trigger_rate']:.4f}"
    )
    logger.info(f"结果保存: {step_results_path}")
