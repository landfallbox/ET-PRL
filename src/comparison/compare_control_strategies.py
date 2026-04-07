from __future__ import annotations

from pathlib import Path

import numpy as np
from ml_toolkit.rl import SequenceEnv
from ml_toolkit.utils import create_experiment_context

from config.control_compare_config import ControlCompareConfig
from src.control_evaluation.common import (
    build_test_components,
    copy_train_config,
    create_streaming_gate,
    resolve_train_experiment_dir,
)
from src.control_evaluation.strategies.event_driven import test_event_driven
from src.control_evaluation.strategies.fixed_interval import test_fixed_interval
from src.control_evaluation.strategies.event_triggered_etc import test_event_triggered_etc
from src.control_evaluation.strategies.pid import test_pid


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


def _resolve_train_dir_from_model_path(dqn_model_path: Path | None) -> Path | None:
    if dqn_model_path is None:
        return None

    resolved_model_path = Path(dqn_model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"指定的 DQN 模型文件不存在: {resolved_model_path}")

    checkpoint_dir = resolved_model_path.parent
    if checkpoint_dir.name != ControlCompareConfig.CHECKPOINT_DIR_NAME:
        raise ValueError(
            "dqn_model_path 的父目录应为 checkpoints 目录。"
            f"当前路径: {resolved_model_path}"
        )

    return checkpoint_dir.parent


def _resolve_event_thresholds(config: type[ControlCompareConfig], test_data) -> tuple[float, float, float]:
    def _attr_or_std(column: str, attr_name: str, fallback: float) -> float:
        config_value = getattr(config, attr_name, None)
        if config_value is not None:
            return max(float(config_value), 1e-6)
        if column in test_data.columns:
            return max(float(test_data[column].std(ddof=0)), 1e-6)
        return float(fallback)

    cl_threshold = _attr_or_std("CL", "EVENT_CL_THRESHOLD", 50.0)
    twb_threshold = _attr_or_std("Twb", "EVENT_TWB_THRESHOLD", 1.0)
    cl_predict_threshold = _attr_or_std("CL_predict", "EVENT_CL_PREDICT_THRESHOLD", 50.0)
    return cl_threshold, twb_threshold, cl_predict_threshold


def compare_control_strategies(
    train_experiment_dir: Path | None = None,
    fixed_interval: int | None = None,
    fixed_intervals: list[int] | None = None,
    gate_state_path: Path | None = None,
    dqn_model_path: Path | None = None,
) -> None:
    config = ControlCompareConfig
    test_experiment_dir = config.get_eval_experiment_dir()

    context = create_experiment_context(
        experiment_dir=test_experiment_dir,
        config=config,
        save_config=False,
        log_filename=config.EVALUATION_LOG_FILENAME,
        metrics_filename=config.EVALUATION_METRICS_FILENAME,
        config_filename=config.CONFIG_FILENAME,
    )
    logger = context.logger
    metrics_recorder = context.metrics_recorder

    if fixed_intervals is not None:
        resolved_fixed_intervals = [int(i) for i in fixed_intervals]
    elif fixed_interval is not None:
        resolved_fixed_intervals = [int(fixed_interval)]
    else:
        resolved_fixed_intervals = [1]

    if not resolved_fixed_intervals:
        raise ValueError("fixed_intervals 不能为空")

    for interval in resolved_fixed_intervals:
        if interval <= 0:
            raise ValueError(f"fixed_interval 必须大于 0，当前: {interval}")

    model_train_dir = _resolve_train_dir_from_model_path(dqn_model_path)
    resolved_train_dir = resolve_train_experiment_dir(model_train_dir or train_experiment_dir)
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

    base_payload = {
        "train_experiment_dir": str(resolved_train_dir),
        "best_epoch_from_train": int(checkpoint.get("epoch", -1)) + 1,
        "best_metrics_from_train": checkpoint.get("metrics", {}),
        "mode": "compare",
        "fixed_intervals": [int(i) for i in resolved_fixed_intervals],
        "dqn_model_path": str(dqn_model_path) if dqn_model_path is not None else None,
        "gate_state_path": str(gate_state_path) if gate_state_path is not None else None,
    }

    summaries: dict[str, dict] = {}

    for interval in resolved_fixed_intervals:
        fixed_key = f"fixed_interval_{interval}"
        fixed_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
        fixed_summary, fixed_step_results = test_fixed_interval(
            agent=agent,
            env=fixed_env,
            action_space=action_space,
            fixed_interval=interval,
            supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
            comfort_lower_bound=config.COMFORT_LOWER_BOUND,
            comfort_upper_bound=config.COMFORT_UPPER_BOUND,
        )
        summaries[fixed_key] = fixed_summary
        fixed_step_results.to_csv(test_experiment_dir / f"{fixed_key}_step_results.csv", index=False)

    event_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
    event_gate = create_streaming_gate(config=config, test_data=test_data, logger=logger, gate_state_path=gate_state_path)
    event_summary, event_step_results = test_event_driven(
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
    summaries["event_driven"] = event_summary
    event_step_results.to_csv(test_experiment_dir / "event_driven_step_results.csv", index=False)

    pid_kp = float(getattr(config, "PID_KP", 0.6))
    pid_ki = float(getattr(config, "PID_KI", 0.05))
    pid_kd = float(getattr(config, "PID_KD", 0.1))
    pid_integral_limit = float(
        getattr(config, "PID_INTEGRAL_LIMIT", max(float(np.max(action_space) - np.min(action_space)) * 4.0, 1.0))
    )
    pid_error_deadband = float(getattr(config, "PID_ERROR_DEADBAND", 0.1))
    pid_derivative_filter_alpha = float(getattr(config, "PID_DERIVATIVE_FILTER_ALPHA", 0.7))
    pid_max_action_step = float(getattr(config, "PID_MAX_ACTION_STEP", 2.0))
    pid_target_power_ratio = float(getattr(config, "PID_TARGET_POWER_RATIO", 0.85))
    pid_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
    pid_summary, pid_step_results = test_pid(
        env=pid_env,
        action_space=action_space,
        supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
        kp=pid_kp,
        ki=pid_ki,
        kd=pid_kd,
        integral_limit=pid_integral_limit,
        error_deadband=pid_error_deadband,
        derivative_filter_alpha=pid_derivative_filter_alpha,
        max_action_step=pid_max_action_step,
        target_power_ratio=pid_target_power_ratio,
    )
    summaries["pid"] = pid_summary
    pid_step_results.to_csv(test_experiment_dir / "pid_step_results.csv", index=False)

    cl_threshold, twb_threshold, cl_predict_threshold = _resolve_event_thresholds(config, test_data)
    event_trigger_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
    event_trigger_summary, event_trigger_step_results = test_event_triggered_etc(
        agent=agent,
        env=event_trigger_env,
        data=test_data,
        action_space=action_space,
        feature_columns=config.FEATURE_COLUMNS,
        supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
        min_delta_thresholds={
            "CL": cl_threshold,
            "Twb": twb_threshold,
            "CL_predict": cl_predict_threshold,
        },
        threshold_window_size=min(int(getattr(config, "GATE_LOCAL_WINDOW_SIZE", 20)), 20),
        threshold_min_samples=min(int(getattr(config, "THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION", 3)), 3),
        trigger_score_threshold=float(getattr(config, "EVENT_TRIGGER_SCORE_THRESHOLD", 0.6)),
    )
    summaries["event_triggered_etc"] = event_trigger_summary
    event_trigger_step_results.to_csv(test_experiment_dir / "event_triggered_etc_step_results.csv", index=False)

    metrics_recorder.save_metrics(
        {
            **base_payload,
            "paper_symbol_mapping": _get_paper_symbol_field_mapping(),
            "strategies_performance": summaries,
            "pid_params": {
                "kp": pid_kp,
                "ki": pid_ki,
                "kd": pid_kd,
                "integral_limit": pid_integral_limit,
                "error_deadband": pid_error_deadband,
                "derivative_filter_alpha": pid_derivative_filter_alpha,
                "max_action_step": pid_max_action_step,
                "target_power_ratio": pid_target_power_ratio,
            },
            "event_threshold_params": {
                "CL": cl_threshold,
                "Twb": twb_threshold,
                "CL_predict": cl_predict_threshold,
            },
        }
    )

    logger.info("控制策略性能指标对比")
    for strategy_name, metrics in summaries.items():
        action_count = metrics.get("action_count", 0)
        energy_total = metrics.get("E_total_kwh", 0)
        comfort_avg = metrics.get("avg_comfort_score")
        comfort_text = f"{float(comfort_avg):.4f}" if comfort_avg is not None else "N/A"
        if strategy_name == "pid":
            primary_label = "energy_kwh"
            primary_value = float(energy_total)
        else:
            primary_label = "reward"
            primary_value = float(metrics.get("total_reward", 0))
        logger.info(
            f"策略: {strategy_name:25s} | {primary_label}={primary_value:8.4f} | "
            f"actions={action_count:5d} | energy={energy_total:8.2f}kWh | comfort={comfort_text}"
        )
