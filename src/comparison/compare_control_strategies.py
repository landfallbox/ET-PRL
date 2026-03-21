from __future__ import annotations

from pathlib import Path

from ml_toolkit.rl import SequenceEnv
from ml_toolkit.utils import create_experiment_context

from config.compare_dqn_config import CompareDQNConfig
from src.control_evaluation.common import (
    build_eval_components,
    copy_train_config,
    create_streaming_gate,
    resolve_train_experiment_dir,
)
from src.control_evaluation.strategies import evaluate_event_driven, evaluate_fixed_interval


def _get_paper_symbol_field_mapping() -> dict[str, str]:
    try:
        from src.control_evaluation.common import get_paper_symbol_field_mapping

        return get_paper_symbol_field_mapping()
    except (ImportError, AttributeError):
        return {
            "E_daily": "E_daily_kwh_per_day",
            "eta_saving": "eta_saving_pct",
            "PPR": "PPR_percent",
            "sigma_delta_a": "sigma_delta_a",
            "N_daily": "N_daily_count_per_day",
            "ACR": "ACR",
        }


def _build_comparison(fixed_summary: dict, event_summary: dict) -> dict:
    fixed_reward = float(fixed_summary.get("total_reward", 0.0))
    event_reward = float(event_summary.get("total_reward", 0.0))

    fixed_actions = int(fixed_summary.get("action_count", 0))
    event_actions = int(event_summary.get("action_count", 0))
    action_reduction = fixed_actions - event_actions
    action_reduction_pct = (action_reduction / fixed_actions * 100.0) if fixed_actions > 0 else 0.0

    ppr = (event_reward / fixed_reward) if fixed_reward != 0 else 0.0
    acr = (fixed_actions / event_actions) if event_actions > 0 else float("inf")
    e_fixed = float(fixed_summary.get("E_total_kwh", 0.0))
    e_event = float(event_summary.get("E_total_kwh", 0.0))
    eta_saving_pct = ((e_fixed - e_event) / e_fixed * 100.0) if e_fixed > 0 else 0.0

    e_daily_fixed = float(fixed_summary.get("E_daily_kwh_per_day", 0.0))
    e_daily_event = float(event_summary.get("E_daily_kwh_per_day", 0.0))
    n_daily_fixed = float(fixed_summary.get("N_daily_count_per_day", 0.0))
    n_daily_event = float(event_summary.get("N_daily_count_per_day", 0.0))
    sigma_delta_a_fixed = float(fixed_summary.get("sigma_delta_a", 0.0))
    sigma_delta_a_event = float(event_summary.get("sigma_delta_a", 0.0))
    trigger_rate_fixed = float(fixed_summary.get("event_trigger_rate", 0.0))
    trigger_rate_event = float(event_summary.get("event_trigger_rate", 0.0))

    return {
        "reward_change": event_reward - fixed_reward,
        "reward_change_pct": ((event_reward - fixed_reward) / fixed_reward * 100.0) if fixed_reward != 0 else 0.0,
        "comfort_change": float(event_summary.get("avg_comfort_score", 0.0))
        - float(fixed_summary.get("avg_comfort_score", 0.0)),
        "energy_change": float(event_summary.get("avg_energy_score", 0.0))
        - float(fixed_summary.get("avg_energy_score", 0.0)),
        "action_reduction": int(action_reduction),
        "action_reduction_pct": float(action_reduction_pct),
        "PPR": float(ppr),
        "PPR_percent": float(ppr * 100.0),
        "ACR": float(acr),
        "eta_saving_pct": float(eta_saving_pct),
        "E_baseline_total_kwh": float(e_fixed),
        "E_ET_PRL_total_kwh": float(e_event),
        "E_baseline_daily_kwh_per_day": float(e_daily_fixed),
        "E_ET_PRL_daily_kwh_per_day": float(e_daily_event),
        "E_daily_change": float(e_daily_event - e_daily_fixed),
        "E_daily_change_pct": float(((e_daily_event - e_daily_fixed) / e_daily_fixed * 100.0) if e_daily_fixed > 0 else 0.0),
        "N_daily_baseline": float(n_daily_fixed),
        "N_daily_ET_PRL": float(n_daily_event),
        "N_daily_change": float(n_daily_event - n_daily_fixed),
        "N_daily_change_pct": float(((n_daily_event - n_daily_fixed) / n_daily_fixed * 100.0) if n_daily_fixed > 0 else 0.0),
        "sigma_delta_a_baseline": float(sigma_delta_a_fixed),
        "sigma_delta_a_ET_PRL": float(sigma_delta_a_event),
        "sigma_delta_a_change": float(sigma_delta_a_event - sigma_delta_a_fixed),
        "sigma_delta_a_change_pct": float(
            ((sigma_delta_a_event - sigma_delta_a_fixed) / sigma_delta_a_fixed * 100.0) if sigma_delta_a_fixed > 0 else 0.0
        ),
        "trigger_rate_baseline": float(trigger_rate_fixed),
        "trigger_rate_ET_PRL": float(trigger_rate_event),
        "trigger_rate_change": float(trigger_rate_event - trigger_rate_fixed),
        "trigger_rate_change_pct": float(
            ((trigger_rate_event - trigger_rate_fixed) / trigger_rate_fixed * 100.0) if trigger_rate_fixed > 0 else 0.0
        ),
    }


def compare_control_strategies(
    train_experiment_dir: Path | None = None,
    fixed_interval: int = 4,
    gate_state_path: Path | None = None,
) -> None:
    config = CompareDQNConfig
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

    base_payload = {
        "train_experiment_dir": str(resolved_train_dir),
        "best_epoch_from_train": int(checkpoint.get("epoch", -1)) + 1,
        "best_metrics_from_train": checkpoint.get("metrics", {}),
        "mode": "compare",
        "fixed_interval": int(fixed_interval),
        "gate_state_path": str(gate_state_path) if gate_state_path is not None else None,
    }

    fixed_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
    fixed_summary, fixed_step_results = evaluate_fixed_interval(
        agent=agent,
        env=fixed_env,
        action_space=action_space,
        fixed_interval=fixed_interval,
        supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
        comfort_lower_bound=config.COMFORT_LOWER_BOUND,
        comfort_upper_bound=config.COMFORT_UPPER_BOUND,
    )

    event_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
    event_gate = create_streaming_gate(config=config, test_data=test_data, logger=logger, gate_state_path=gate_state_path)
    event_summary, event_step_results = evaluate_event_driven(
        agent=agent,
        env=event_env,
        data=test_data,
        action_space=action_space,
        gate=event_gate,
        feature_columns=config.FEATURE_COLUMNS,
        supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
        comfort_lower_bound=config.COMFORT_LOWER_BOUND,
        comfort_upper_bound=config.COMFORT_UPPER_BOUND,
    )

    comparison = _build_comparison(fixed_summary=fixed_summary, event_summary=event_summary)
    metrics_recorder.save_metrics(
        {
            **base_payload,
            "paper_symbol_mapping": _get_paper_symbol_field_mapping(),
            "fixed_interval_summary": fixed_summary,
            "event_driven_summary": event_summary,
            "comparison": comparison,
        }
    )

    fixed_step_results_path = eval_experiment_dir / "fixed_interval_step_results.csv"
    event_step_results_path = eval_experiment_dir / "event_driven_step_results.csv"
    fixed_step_results.to_csv(fixed_step_results_path, index=False)
    event_step_results.to_csv(event_step_results_path, index=False)

    logger.info("评估完成: fixed_interval vs event_driven")
    logger.info(
        "固定间隔: "
        f"total_reward={fixed_summary['total_reward']:.4f}, "
        f"action_count={fixed_summary['action_count']}"
    )
    logger.info(
        "事件驱动: "
        f"total_reward={event_summary['total_reward']:.4f}, "
        f"action_count={event_summary['action_count']}, "
        f"event_trigger_rate={event_summary['event_trigger_rate']:.4f}"
    )
    logger.info(
        "对比指标: "
        f"PPR={comparison['PPR']:.4f}, "
        f"ACR={comparison['ACR']:.4f}, "
        f"action_reduction_pct={comparison['action_reduction_pct']:.2f}%"
    )
