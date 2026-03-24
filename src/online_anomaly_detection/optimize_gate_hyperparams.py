from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ml_toolkit.rl import SequenceEnv
from ml_toolkit.utils import BayesianOptimizer, HyperparameterSpace, Logger

from config.compare_dqn_config import CompareDQNConfig
from config.online_anomaly_detection_config import OnlineAnomalyDetectionConfig
from src.control_evaluation.common import (
    build_eval_components,
    create_streaming_gate,
    resolve_train_experiment_dir,
)
from src.control_evaluation.strategies.event_driven import evaluate_event_driven
from src.control_evaluation.strategies.fixed_interval import evaluate_fixed_interval


def _load_prewarm_features(
    config: type[CompareDQNConfig],
    val_df_override: pd.DataFrame | None = None,
) -> np.ndarray:
    feature_columns = list(config.FEATURE_COLUMNS)

    train_df = pd.read_csv(config.get_train_data_path())
    val_df = val_df_override if val_df_override is not None else pd.read_csv(config.get_val_data_path())

    missing_train = [column for column in feature_columns if column not in train_df.columns]
    missing_val = [column for column in feature_columns if column not in val_df.columns]
    if missing_train:
        raise ValueError(f"训练集缺少门控特征列: {missing_train}")
    if missing_val:
        raise ValueError(f"验证集缺少门控特征列: {missing_val}")

    merged = pd.concat([train_df[feature_columns], val_df[feature_columns]], ignore_index=True)
    return merged.to_numpy(dtype=np.float32)


def _create_search_space(config: type[CompareDQNConfig]) -> HyperparameterSpace:
    space = HyperparameterSpace()
    (
        space.add_float(
            "global_ema_decay",
            config.GATE_OPT_GLOBAL_EMA_DECAY_MIN,
            config.GATE_OPT_GLOBAL_EMA_DECAY_MAX,
        )
        .add_float(
            "alpha_local_weight",
            config.GATE_OPT_ALPHA_LOCAL_WEIGHT_MIN,
            config.GATE_OPT_ALPHA_LOCAL_WEIGHT_MAX,
        )
        .add_int(
            "local_window_size",
            config.GATE_OPT_LOCAL_WINDOW_SIZE_MIN,
            config.GATE_OPT_LOCAL_WINDOW_SIZE_MAX,
        )
        .add_int(
            "reference_samples",
            config.GATE_OPT_REFERENCE_SAMPLES_MIN,
            config.GATE_OPT_REFERENCE_SAMPLES_MAX,
        )
        .add_float(
            "contamination",
            config.GATE_OPT_CONTAMINATION_MIN,
            config.GATE_OPT_CONTAMINATION_MAX,
        )
        .add_float(
            "threshold_bias",
            config.GATE_OPT_THRESHOLD_BIAS_MIN,
            config.GATE_OPT_THRESHOLD_BIAS_MAX,
        )
        .add_float(
            "threshold_quantile",
            config.GATE_OPT_THRESHOLD_QUANTILE_MIN,
            config.GATE_OPT_THRESHOLD_QUANTILE_MAX,
        )
        .add_float(
            "threshold_mad_scale",
            config.GATE_OPT_THRESHOLD_MAD_SCALE_MIN,
            config.GATE_OPT_THRESHOLD_MAD_SCALE_MAX,
        )
        .add_float(
            "threshold_local_update_rate",
            config.GATE_OPT_THRESHOLD_LOCAL_UPDATE_RATE_MIN,
            config.GATE_OPT_THRESHOLD_LOCAL_UPDATE_RATE_MAX,
        )
        .add_float(
            "threshold_quantile_weight",
            config.GATE_OPT_THRESHOLD_QUANTILE_WEIGHT_MIN,
            config.GATE_OPT_THRESHOLD_QUANTILE_WEIGHT_MAX,
        )
        .add_int(
            "threshold_min_samples_for_optimization",
            config.GATE_OPT_THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION_MIN,
            config.GATE_OPT_THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION_MAX,
        )
        .add_float(
            "score_short_weight",
            config.GATE_OPT_SCORE_SHORT_WEIGHT_MIN,
            config.GATE_OPT_SCORE_SHORT_WEIGHT_MAX,
        )
        .add_float(
            "score_medium_weight",
            config.GATE_OPT_SCORE_MEDIUM_WEIGHT_MIN,
            config.GATE_OPT_SCORE_MEDIUM_WEIGHT_MAX,
        )
        .add_float(
            "trigger_hysteresis_margin",
            config.GATE_OPT_TRIGGER_HYSTERESIS_MARGIN_MIN,
            config.GATE_OPT_TRIGGER_HYSTERESIS_MARGIN_MAX,
        )
        .add_int(
            "min_trigger_interval_steps",
            config.GATE_OPT_MIN_TRIGGER_INTERVAL_STEPS_MIN,
            config.GATE_OPT_MIN_TRIGGER_INTERVAL_STEPS_MAX,
        )
    )
    return space


def _build_trial_config(base_cls: type[CompareDQNConfig], params: dict) -> type[CompareDQNConfig]:
    class TrialConfig(base_cls):
        pass

    setattr(TrialConfig, "GATE_GLOBAL_EMA_DECAY", float(params["global_ema_decay"]))
    setattr(TrialConfig, "GATE_ALPHA_LOCAL_WEIGHT", float(params["alpha_local_weight"]))
    setattr(TrialConfig, "GATE_LOCAL_WINDOW_SIZE", int(params["local_window_size"]))
    setattr(TrialConfig, "GATE_REFERENCE_SAMPLES", int(params["reference_samples"]))
    setattr(TrialConfig, "GATE_CONTAMINATION", float(params["contamination"]))

    setattr(TrialConfig, "GATE_THRESHOLD_BIAS", float(params["threshold_bias"]))
    setattr(TrialConfig, "THRESHOLD_QUANTILE", float(params["threshold_quantile"]))
    setattr(TrialConfig, "THRESHOLD_MAD_SCALE", float(params["threshold_mad_scale"]))
    setattr(TrialConfig, "THRESHOLD_LOCAL_UPDATE_RATE", float(params["threshold_local_update_rate"]))
    setattr(TrialConfig, "THRESHOLD_QUANTILE_WEIGHT", float(params["threshold_quantile_weight"]))
    setattr(
        TrialConfig,
        "THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION",
        int(params["threshold_min_samples_for_optimization"]),
    )

    score_short_weight = float(params["score_short_weight"])
    score_medium_weight = float(params["score_medium_weight"])
    score_long_weight = max(0.01, 1.0 - score_short_weight - score_medium_weight)

    setattr(TrialConfig, "GATE_SCORE_SHORT_WEIGHT", score_short_weight)
    setattr(TrialConfig, "GATE_SCORE_MEDIUM_WEIGHT", score_medium_weight)
    setattr(TrialConfig, "GATE_SCORE_LONG_WEIGHT", float(score_long_weight))
    setattr(TrialConfig, "GATE_TRIGGER_HYSTERESIS_MARGIN", float(params["trigger_hysteresis_margin"]))
    setattr(TrialConfig, "GATE_MIN_TRIGGER_INTERVAL_STEPS", int(params["min_trigger_interval_steps"]))

    return TrialConfig


def _calculate_recall_metrics(step_results: pd.DataFrame) -> dict:
    violation = (
        (step_results["chiller_supply_temp"] < 15.0) | (step_results["chiller_supply_temp"] > 19.0)
    ).astype(int)
    gate_signal = step_results["gate_signal"].astype(int)

    tp = int(((gate_signal == 1) & (violation == 1)).sum())
    fp = int(((gate_signal == 1) & (violation == 0)).sum())
    fn = int(((gate_signal == 0) & (violation == 1)).sum())
    tn = int(((gate_signal == 0) & (violation == 0)).sum())

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    false_discovery_rate = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    action_rate = float(step_results["action_updated"].mean()) if not step_results.empty else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": recall,
        "precision": precision,
        "false_discovery_rate": false_discovery_rate,
        "false_positive_rate": false_positive_rate,
        "action_rate": action_rate,
    }


def _positive_part(value: float) -> float:
    return float(max(0.0, value))


def _split_validation_for_optimization(
    val_data: pd.DataFrame,
    fit_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None, float]:
    fit_ratio = float(np.clip(fit_ratio, 0.5, 0.95))

    if len(val_data) < 100:
        return val_data.reset_index(drop=True), None, 1.0

    split_index = int(len(val_data) * fit_ratio)
    split_index = int(np.clip(split_index, 1, len(val_data) - 1))

    fit_data = val_data.iloc[:split_index].reset_index(drop=True)
    holdout_data = val_data.iloc[split_index:].reset_index(drop=True)
    if holdout_data.empty:
        return fit_data, None, 1.0

    realized_ratio = float(len(fit_data) / max(len(val_data), 1))
    return fit_data, holdout_data, realized_ratio


def optimize_gate_hyperparameters(
    train_experiment_dir: Path | None = None,
    n_trials: int | None = None,
    n_jobs: int | None = None,
) -> dict:
    base_config = CompareDQNConfig

    if n_trials is None:
        n_trials = int(base_config.GATE_OPTIMIZATION_DEFAULT_TRIALS)
    if n_jobs is None:
        n_jobs = int(base_config.GATE_OPTIMIZATION_DEFAULT_N_JOBS)

    output_dir = OnlineAnomalyDetectionConfig.get_optimization_dir() / OnlineAnomalyDetectionConfig.TIMESTAMP / "gate"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(output_dir)
    logger.info("开始 Gate 超参优化（最小动作频率 + 性能退化约束惩罚）")
    logger.info(f"试验次数: {n_trials}, 并行任务数: {n_jobs}")

    reward_drop_tolerance_ratio = float(base_config.GATE_OPT_REWARD_DROP_TOLERANCE_RATIO)
    energy_increase_tolerance_ratio = float(base_config.GATE_OPT_ENERGY_INCREASE_TOLERANCE_RATIO)
    reward_drop_penalty_weight = float(base_config.GATE_OPT_REWARD_DROP_PENALTY_WEIGHT)
    energy_increase_penalty_weight = float(base_config.GATE_OPT_ENERGY_INCREASE_PENALTY_WEIGHT)
    baseline_fixed_interval = int(base_config.GATE_OPT_BASELINE_FIXED_INTERVAL)
    validation_fit_ratio = float(getattr(base_config, "GATE_OPT_VALIDATION_FIT_RATIO", 0.8))

    logger.info(
        "目标函数: objective = action_rate + reward_penalty + energy_penalty"
    )
    logger.info(
        "其中: reward_penalty = "
        f"{reward_drop_penalty_weight:.3f}*(max(0,reward_drop_ratio-{reward_drop_tolerance_ratio:.4f})/{max(reward_drop_tolerance_ratio, 1e-8):.4f})^2"
    )
    logger.info(
        "其中: energy_penalty = "
        f"{energy_increase_penalty_weight:.3f}*(max(0,energy_increase_ratio-{energy_increase_tolerance_ratio:.4f})/{max(energy_increase_tolerance_ratio, 1e-8):.4f})^2"
    )

    resolved_train_dir = resolve_train_experiment_dir(train_experiment_dir)
    logger.info(f"使用训练实验目录: {resolved_train_dir}")

    val_data, action_space, agent, checkpoint, reward_calc = build_eval_components(
        config=base_config,
        resolved_train_dir=resolved_train_dir,
        data_split="val",
    )
    fit_data, holdout_data, realized_fit_ratio = _split_validation_for_optimization(
        val_data=val_data,
        fit_ratio=validation_fit_ratio,
    )
    prewarm_features = _load_prewarm_features(base_config, val_df_override=fit_data)

    logger.info(
        "验证集时间切分: "
        f"fit={len(fit_data)}步 ({realized_fit_ratio:.2%}), "
        f"holdout={len(holdout_data) if holdout_data is not None else 0}步"
    )

    baseline_env = SequenceEnv(fit_data, base_config.STATE_COLUMNS, reward_calc)
    baseline_summary, _ = evaluate_fixed_interval(
        agent=agent,
        env=baseline_env,
        action_space=action_space,
        fixed_interval=baseline_fixed_interval,
        supply_temp_ref=base_config.CHILLER_SUPPLY_TEMP_REF,
        comfort_lower_bound=base_config.COMFORT_LOWER_BOUND,
        comfort_upper_bound=base_config.COMFORT_UPPER_BOUND,
    )
    baseline_total_reward = float(baseline_summary["total_reward"])
    baseline_energy_daily = float(baseline_summary["E_daily_kwh_per_day"])
    baseline_action_rate = float(baseline_summary["action_frequency"])
    logger.info(
        "验证集基线(fixed_interval="
        f"{baseline_fixed_interval}): reward={baseline_total_reward:.2f}, "
        f"energy_daily={baseline_energy_daily:.4f}, action_rate={baseline_action_rate:.4f}"
    )

    space = _create_search_space(base_config)
    optimizer = BayesianOptimizer(
        space=space,
        output_dir=output_dir,
        sampler=str(base_config.GATE_OPTIMIZATION_SAMPLER),
        seed=int(base_config.GATE_OPTIMIZATION_SEED),
    )

    active_trials = {"count": 0}
    active_lock = threading.Lock()

    def objective(trial, params: dict) -> float:
        start_time = time.perf_counter()
        worker_name = threading.current_thread().name
        with active_lock:
            active_trials["count"] += 1
            current_active = active_trials["count"]

        logger.info(
            f"Trial {trial.number} 开始 | worker={worker_name} | active_trials={current_active} | "
            f"alpha={params['alpha_local_weight']:.3f}, ema={params['global_ema_decay']:.4f}, "
            f"q={params['threshold_quantile']:.3f}, qw={params['threshold_quantile_weight']:.3f}, "
            f"bias={params['threshold_bias']:.3f}, "
            f"cont={params['contamination']:.3f}, ref={params['reference_samples']}, "
            f"delta_min={params['min_trigger_interval_steps']}"
        )

        try:
            trial_config = _build_trial_config(base_config, params)

            gate = create_streaming_gate(
                config=trial_config,
                test_data=fit_data,
                logger=logger,
                gate_state_path=None,
            )
            for sample in prewarm_features:
                gate.predict(sample)

            env = SequenceEnv(fit_data, trial_config.STATE_COLUMNS, reward_calc)
            summary, step_results = evaluate_event_driven(
                agent=agent,
                env=env,
                data=fit_data,
                action_space=action_space,
                gate=gate,
                feature_columns=trial_config.FEATURE_COLUMNS,
                supply_temp_ref=trial_config.CHILLER_SUPPLY_TEMP_REF,
            )

            metrics = _calculate_recall_metrics(step_results)
            recall = float(metrics["recall"])
            false_discovery_rate = float(metrics["false_discovery_rate"])
            action_rate = float(metrics["action_rate"])
            total_reward = float(summary["total_reward"])
            energy_daily = float(summary["E_daily_kwh_per_day"])

            reward_drop_ratio = _positive_part((baseline_total_reward - total_reward) / max(abs(baseline_total_reward), 1e-8))
            energy_increase_ratio = _positive_part((energy_daily - baseline_energy_daily) / max(abs(baseline_energy_daily), 1e-8))

            reward_violation_ratio = _positive_part(reward_drop_ratio - reward_drop_tolerance_ratio) / max(
                reward_drop_tolerance_ratio, 1e-8
            )
            energy_violation_ratio = _positive_part(
                energy_increase_ratio - energy_increase_tolerance_ratio
            ) / max(energy_increase_tolerance_ratio, 1e-8)

            reward_penalty = reward_drop_penalty_weight * (reward_violation_ratio**2)
            energy_penalty = energy_increase_penalty_weight * (energy_violation_ratio**2)

            objective_score = action_rate + reward_penalty + energy_penalty

            trial.set_user_attr("recall", recall)
            trial.set_user_attr("precision", float(metrics["precision"]))
            trial.set_user_attr("false_discovery_rate", false_discovery_rate)
            trial.set_user_attr("false_positive_rate", float(metrics["false_positive_rate"]))
            trial.set_user_attr("action_rate", action_rate)
            trial.set_user_attr("total_reward", total_reward)
            trial.set_user_attr("energy_daily_kwh", energy_daily)
            trial.set_user_attr("reward_drop_ratio", reward_drop_ratio)
            trial.set_user_attr("energy_increase_ratio", energy_increase_ratio)
            trial.set_user_attr("reward_penalty", reward_penalty)
            trial.set_user_attr("energy_penalty", energy_penalty)
            trial.set_user_attr("objective_score", objective_score)

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Trial {trial.number} 完成: objective={objective_score:.6f}, recall={recall:.6f}, "
                f"fdr={false_discovery_rate:.6f}, precision={metrics['precision']:.6f}, "
                f"action_rate={action_rate:.4f}, reward_penalty={reward_penalty:.6f}, "
                f"energy_penalty={energy_penalty:.6f}, "
                f"total_reward={total_reward:.2f}, "
                f"elapsed={elapsed:.1f}s"
            )

            return objective_score

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Trial {trial.number} 失败: {exc}")
            logger.info(f"Trial {trial.number} 结束(失败) | worker={worker_name} | elapsed={elapsed:.1f}s")
            return 1e9

        finally:
            with active_lock:
                active_trials["count"] -= 1
                current_active = active_trials["count"]
            logger.info(f"Trial {trial.number} 释放 worker={worker_name} | active_trials={current_active}")

    result = optimizer.optimize(
        objective_fn=objective,
        n_trials=int(n_trials),
        n_jobs=int(n_jobs),
    )

    best_objective_score = float(result["best_value"])
    best_params = result["best_params"]
    best_trial = optimizer.study.best_trial if optimizer.study is not None else None
    best_recall = float(best_trial.user_attrs.get("recall", 0.0)) if best_trial is not None else 0.0
    best_precision = float(best_trial.user_attrs.get("precision", 0.0)) if best_trial is not None else 0.0
    best_fdr = float(best_trial.user_attrs.get("false_discovery_rate", 0.0)) if best_trial is not None else 0.0
    best_action_rate = float(best_trial.user_attrs.get("action_rate", 0.0)) if best_trial is not None else 0.0
    best_config_overrides = {
        "GATE_GLOBAL_EMA_DECAY": float(best_params["global_ema_decay"]),
        "GATE_ALPHA_LOCAL_WEIGHT": float(best_params["alpha_local_weight"]),
        "GATE_LOCAL_WINDOW_SIZE": int(best_params["local_window_size"]),
        "GATE_REFERENCE_SAMPLES": int(best_params["reference_samples"]),
        "GATE_CONTAMINATION": float(best_params["contamination"]),
        "GATE_THRESHOLD_BIAS": float(best_params["threshold_bias"]),
        "THRESHOLD_QUANTILE": float(best_params["threshold_quantile"]),
        "THRESHOLD_MAD_SCALE": float(best_params["threshold_mad_scale"]),
        "THRESHOLD_LOCAL_UPDATE_RATE": float(best_params["threshold_local_update_rate"]),
        "THRESHOLD_QUANTILE_WEIGHT": float(best_params["threshold_quantile_weight"]),
        "THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION": int(best_params["threshold_min_samples_for_optimization"]),
        "GATE_SCORE_SHORT_WEIGHT": float(best_params["score_short_weight"]),
        "GATE_SCORE_MEDIUM_WEIGHT": float(best_params["score_medium_weight"]),
        "GATE_SCORE_LONG_WEIGHT": float(
            max(0.01, 1.0 - float(best_params["score_short_weight"]) - float(best_params["score_medium_weight"]))
        ),
        "GATE_TRIGGER_HYSTERESIS_MARGIN": float(best_params["trigger_hysteresis_margin"]),
        "GATE_MIN_TRIGGER_INTERVAL_STEPS": int(best_params["min_trigger_interval_steps"]),
    }

    holdout_evaluation: dict | None = None
    if holdout_data is not None and not holdout_data.empty:
        best_trial_config = _build_trial_config(base_config, best_params)

        holdout_baseline_env = SequenceEnv(holdout_data, base_config.STATE_COLUMNS, reward_calc)
        holdout_baseline_summary, _ = evaluate_fixed_interval(
            agent=agent,
            env=holdout_baseline_env,
            action_space=action_space,
            fixed_interval=baseline_fixed_interval,
            supply_temp_ref=base_config.CHILLER_SUPPLY_TEMP_REF,
            comfort_lower_bound=base_config.COMFORT_LOWER_BOUND,
            comfort_upper_bound=base_config.COMFORT_UPPER_BOUND,
        )

        holdout_gate = create_streaming_gate(
            config=best_trial_config,
            test_data=holdout_data,
            logger=logger,
            gate_state_path=None,
        )
        for sample in prewarm_features:
            holdout_gate.predict(sample)

        holdout_env = SequenceEnv(holdout_data, best_trial_config.STATE_COLUMNS, reward_calc)
        holdout_summary, holdout_step_results = evaluate_event_driven(
            agent=agent,
            env=holdout_env,
            data=holdout_data,
            action_space=action_space,
            gate=holdout_gate,
            feature_columns=best_trial_config.FEATURE_COLUMNS,
            supply_temp_ref=best_trial_config.CHILLER_SUPPLY_TEMP_REF,
        )

        holdout_metrics = _calculate_recall_metrics(holdout_step_results)
        holdout_baseline_reward = float(holdout_baseline_summary["total_reward"])
        holdout_baseline_energy_daily = float(holdout_baseline_summary["E_daily_kwh_per_day"])
        holdout_total_reward = float(holdout_summary["total_reward"])
        holdout_energy_daily = float(holdout_summary["E_daily_kwh_per_day"])
        holdout_action_rate = float(holdout_metrics["action_rate"])

        holdout_reward_drop_ratio = _positive_part(
            (holdout_baseline_reward - holdout_total_reward) / max(abs(holdout_baseline_reward), 1e-8)
        )
        holdout_energy_increase_ratio = _positive_part(
            (holdout_energy_daily - holdout_baseline_energy_daily) / max(abs(holdout_baseline_energy_daily), 1e-8)
        )

        holdout_reward_violation_ratio = _positive_part(
            holdout_reward_drop_ratio - reward_drop_tolerance_ratio
        ) / max(reward_drop_tolerance_ratio, 1e-8)
        holdout_energy_violation_ratio = _positive_part(
            holdout_energy_increase_ratio - energy_increase_tolerance_ratio
        ) / max(energy_increase_tolerance_ratio, 1e-8)

        holdout_reward_penalty = reward_drop_penalty_weight * (holdout_reward_violation_ratio**2)
        holdout_energy_penalty = energy_increase_penalty_weight * (holdout_energy_violation_ratio**2)
        holdout_objective_score = holdout_action_rate + holdout_reward_penalty + holdout_energy_penalty

        holdout_evaluation = {
            "samples": int(len(holdout_data)),
            "baseline": {
                "total_reward": holdout_baseline_reward,
                "E_daily_kwh_per_day": holdout_baseline_energy_daily,
                "action_rate": float(holdout_baseline_summary.get("action_frequency", 0.0)),
            },
            "best_trial_result": {
                "objective_score": float(holdout_objective_score),
                "action_rate": holdout_action_rate,
                "recall": float(holdout_metrics["recall"]),
                "precision": float(holdout_metrics["precision"]),
                "false_discovery_rate": float(holdout_metrics["false_discovery_rate"]),
                "reward_drop_ratio": float(holdout_reward_drop_ratio),
                "energy_increase_ratio": float(holdout_energy_increase_ratio),
            },
            "generalization_gap": {
                "objective_score": float(holdout_objective_score - best_objective_score),
                "action_rate": float(holdout_action_rate - best_action_rate),
                "recall": float(holdout_metrics["recall"] - best_recall),
                "precision": float(holdout_metrics["precision"] - best_precision),
                "false_discovery_rate": float(holdout_metrics["false_discovery_rate"] - best_fdr),
            },
        }

        logger.info(
            "Holdout 复评: "
            f"objective={holdout_objective_score:.6f}, action_rate={holdout_action_rate:.6f}, "
            f"recall={holdout_metrics['recall']:.6f}, precision={holdout_metrics['precision']:.6f}, "
            f"fdr={holdout_metrics['false_discovery_rate']:.6f}, "
            f"gap={holdout_objective_score - best_objective_score:+.6f}"
        )

    summary = {
        "train_experiment_dir": str(resolved_train_dir),
        "optimization_data_split": "val_temporal_fit",
        "validation_split": {
            "fit_ratio_requested": validation_fit_ratio,
            "fit_ratio_realized": realized_fit_ratio,
            "fit_samples": int(len(fit_data)),
            "holdout_samples": int(len(holdout_data)) if holdout_data is not None else 0,
        },
        "base_best_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "n_trials": int(result["n_trials"]),
        "baseline": {
            "strategy": "fixed_interval",
            "fixed_interval": baseline_fixed_interval,
            "total_reward": baseline_total_reward,
            "E_daily_kwh_per_day": baseline_energy_daily,
            "action_rate": baseline_action_rate,
        },
        "objective": {
            "type": "min_action_rate_with_performance_guardrails",
            "reward_drop_tolerance_ratio": reward_drop_tolerance_ratio,
            "energy_increase_tolerance_ratio": energy_increase_tolerance_ratio,
            "reward_drop_penalty_weight": reward_drop_penalty_weight,
            "energy_increase_penalty_weight": energy_increase_penalty_weight,
        },
        "best_objective_score": best_objective_score,
        "best_action_rate": best_action_rate,
        "best_recall": best_recall,
        "best_precision": best_precision,
        "best_false_discovery_rate": best_fdr,
        "best_params": best_params,
        "best_config_overrides": best_config_overrides,
        "holdout_evaluation": holdout_evaluation,
    }

    summary_path = output_dir / "best_gate_config.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    if optimizer.study is not None:
        records = []
        for trial in optimizer.study.trials:
            records.append(
                {
                    "trial_number": trial.number,
                    "state": trial.state.name,
                    "objective_value": trial.value,
                    "objective_score": trial.user_attrs.get("objective_score"),
                    "recall": trial.user_attrs.get("recall"),
                    "precision": trial.user_attrs.get("precision"),
                    "false_discovery_rate": trial.user_attrs.get("false_discovery_rate"),
                    "false_positive_rate": trial.user_attrs.get("false_positive_rate"),
                    "action_rate": trial.user_attrs.get("action_rate"),
                    "reward_drop_ratio": trial.user_attrs.get("reward_drop_ratio"),
                    "energy_increase_ratio": trial.user_attrs.get("energy_increase_ratio"),
                    "reward_penalty": trial.user_attrs.get("reward_penalty"),
                    "energy_penalty": trial.user_attrs.get("energy_penalty"),
                    "total_reward": trial.user_attrs.get("total_reward"),
                    "energy_daily_kwh": trial.user_attrs.get("energy_daily_kwh"),
                    **trial.params,
                }
            )
        pd.DataFrame(records).to_csv(output_dir / "gate_trials.csv", index=False)

    logger.info(
        f"优化完成，最优 objective={best_objective_score:.6f}, action_rate={best_action_rate:.6f}, "
        f"recall={best_recall:.6f}, precision={best_precision:.6f}, fdr={best_fdr:.6f}"
    )
    logger.info(f"最优配置已保存: {summary_path}")

    return summary
