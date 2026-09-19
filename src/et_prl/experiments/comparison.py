from __future__ import annotations

from pathlib import Path

import numpy as np

from et_prl.config.loader import load_config
from et_prl.environments import SequenceEnv
from et_prl.evaluation.control.common import (
    build_test_components,
    copy_train_config,
    create_streaming_gate,
    get_paper_symbol_field_mapping,
    resolve_train_experiment_dir,
)
from et_prl.evaluation.control.strategies.event_driven import run_event_driven
from et_prl.evaluation.control.strategies.event_triggered_etc import run_event_triggered_etc
from et_prl.evaluation.control.strategies.fixed_interval import run_fixed_interval
from et_prl.evaluation.control.strategies.pid import run_pid
from et_prl.evaluation.control.strategies.rbc import run_rule_based_control
from et_prl.utils import create_experiment_context

# PID 基线控制器与事件触发基线的固定参数。
# 这些是基线策略的固有参数，不属于可配置超参（不在配置 schema 内），
# 故以模块常量显式声明，避免用 getattr(config, key, default) 伪装成可配置项。
_PID_KP = 0.6
_PID_KI = 0.05
_PID_KD = 0.1
_PID_ERROR_DEADBAND = 0.1
_PID_DERIVATIVE_FILTER_ALPHA = 0.7
_PID_MAX_ACTION_STEP = 2.0
_EVENT_TRIGGER_SCORE_THRESHOLD = 0.6


def _resolve_train_dir_from_model_path(dqn_model_path: Path | None, config) -> Path | None:
    if dqn_model_path is None:
        return None

    resolved_model_path = Path(dqn_model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"指定的 DQN 模型文件不存在: {resolved_model_path}")

    checkpoint_dir = resolved_model_path.parent
    if checkpoint_dir.name != config.CHECKPOINT_DIR_NAME:
        raise ValueError(
            f"dqn_model_path 的父目录应为 checkpoints 目录。当前路径: {resolved_model_path}"
        )

    return checkpoint_dir.parent


def _resolve_event_thresholds(test_data) -> tuple[float, float, float]:
    def _std_or_fallback(column: str, fallback: float) -> float:
        if column in test_data.columns:
            return max(float(test_data[column].std(ddof=0)), 1e-6)
        return float(fallback)

    cl_threshold = _std_or_fallback("CL", 50.0)
    twb_threshold = _std_or_fallback("Twb", 1.0)
    cl_predict_threshold = _std_or_fallback("CL_predict", 50.0)
    return cl_threshold, twb_threshold, cl_predict_threshold


def compare_control_strategies(
    train_experiment_dir: Path | None = None,
    fixed_interval: int | None = None,
    fixed_intervals: list[int] | None = None,
    gate_state_path: Path | None = None,
    dqn_model_path: Path | None = None,
    config=None,
) -> None:
    if config is None:
        config = load_config("control_compare")
    test_experiment_dir = config.get_eval_experiment_dir()

    with create_experiment_context(
        experiment_dir=test_experiment_dir,
        config=config,
        save_config=False,
        log_filename=config.RUN_LOG_FILENAME,
        config_filename=config.CONFIG_FILENAME,
        results_dir=config.get_run_results_dir(mode="eval"),
        tb_dir=config.get_run_tb_dir(mode="eval"),
    ) as context:
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

        model_train_dir = _resolve_train_dir_from_model_path(dqn_model_path, config)
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
            fixed_summary, fixed_step_results = run_fixed_interval(
                agent=agent,
                env=fixed_env,
                action_space=action_space,
                fixed_interval=interval,
                supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
            )
            summaries[fixed_key] = fixed_summary
            fixed_step_results.to_csv(
                config.get_run_results_dir(mode="eval") / f"{fixed_key}_step_results.csv",
                index=False,
            )

        event_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
        event_gate = create_streaming_gate(
            config=config,
            test_data=test_data,
            logger=logger,
            gate_state_path=gate_state_path,
        )
        event_summary, event_step_results = run_event_driven(
            agent=agent,
            env=event_env,
            action_space=action_space,
            gate=event_gate,
            feature_columns=config.FEATURE_COLUMNS,
            supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
        )
        summaries["event_driven"] = event_summary
        event_step_results.to_csv(
            config.get_run_results_dir(mode="eval") / "event_driven_step_results.csv", index=False
        )

        pid_kp = float(_PID_KP)
        pid_ki = float(_PID_KI)
        pid_kd = float(_PID_KD)
        pid_integral_limit = float(
            max(float(np.max(action_space) - np.min(action_space)) * 4.0, 1.0)
        )
        pid_error_deadband = float(_PID_ERROR_DEADBAND)
        pid_derivative_filter_alpha = float(_PID_DERIVATIVE_FILTER_ALPHA)
        pid_max_action_step = float(_PID_MAX_ACTION_STEP)
        pid_supply_temp_ref = float(config.CHILLER_SUPPLY_TEMP_REF)
        pid_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
        pid_summary, pid_step_results = run_pid(
            env=pid_env,
            action_space=action_space,
            supply_temp_ref=pid_supply_temp_ref,
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            integral_limit=pid_integral_limit,
            error_deadband=pid_error_deadband,
            derivative_filter_alpha=pid_derivative_filter_alpha,
            max_action_step=pid_max_action_step,
        )
        summaries["pid"] = pid_summary
        pid_step_results.to_csv(
            config.get_run_results_dir(mode="eval") / "pid_step_results.csv", index=False
        )

        rbc_fixed_setpoint = float(config.RBC_FIXED_SETPOINT)
        rbc_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
        rbc_summary, rbc_step_results = run_rule_based_control(
            env=rbc_env,
            action_space=action_space,
            fixed_setpoint=rbc_fixed_setpoint,
        )
        summaries["rbc"] = rbc_summary
        rbc_step_results.to_csv(
            config.get_run_results_dir(mode="eval") / "rbc_step_results.csv", index=False
        )

        cl_threshold, twb_threshold, cl_predict_threshold = _resolve_event_thresholds(test_data)
        event_trigger_env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
        event_trigger_summary, event_trigger_step_results = run_event_triggered_etc(
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
            threshold_window_size=min(int(config.GATE_LOCAL_WINDOW_SIZE), 20),
            threshold_min_samples=min(int(config.THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION), 3),
            trigger_score_threshold=float(_EVENT_TRIGGER_SCORE_THRESHOLD),
        )
        summaries["event_triggered_etc"] = event_trigger_summary
        event_trigger_step_results.to_csv(
            config.get_run_results_dir(mode="eval") / "event_triggered_etc_step_results.csv",
            index=False,
        )

        metrics_recorder.save_metrics(
            {
                **base_payload,
                "paper_symbol_mapping": get_paper_symbol_field_mapping(),
                "strategies_performance": summaries,
                "pid_params": {
                    "kp": pid_kp,
                    "ki": pid_ki,
                    "kd": pid_kd,
                    "integral_limit": pid_integral_limit,
                    "error_deadband": pid_error_deadband,
                    "derivative_filter_alpha": pid_derivative_filter_alpha,
                    "max_action_step": pid_max_action_step,
                    "supply_temp_ref": pid_supply_temp_ref,
                },
                "rbc_params": {
                    "fixed_setpoint": rbc_fixed_setpoint,
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
            if strategy_name in {"pid", "rbc"}:
                primary_label = "energy_kwh"
                primary_value = float(energy_total)
            else:
                primary_label = "reward"
                primary_value = float(metrics.get("total_reward", 0))
            logger.info(
                f"策略: {strategy_name:25s} | {primary_label}={primary_value:8.4f} | "
                f"actions={action_count:5d} | energy={energy_total:8.2f}kWh | "
                f"comfort={comfort_text}"
            )
