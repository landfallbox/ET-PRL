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


def _load_prewarm_features(config: type[CompareDQNConfig]) -> np.ndarray:
    feature_columns = list(config.FEATURE_COLUMNS)

    train_df = pd.read_csv(config.get_train_data_path())
    val_df = pd.read_csv(config.get_val_data_path())

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
            "alpha_local_weight",
            config.GATE_OPT_ALPHA_LOCAL_WEIGHT_MIN,
            config.GATE_OPT_ALPHA_LOCAL_WEIGHT_MAX,
        )
        .add_int(
            "local_window_size",
            config.GATE_OPT_LOCAL_WINDOW_SIZE_MIN,
            config.GATE_OPT_LOCAL_WINDOW_SIZE_MAX,
        )
        .add_float(
            "global_ema_decay",
            config.GATE_OPT_GLOBAL_EMA_DECAY_MIN,
            config.GATE_OPT_GLOBAL_EMA_DECAY_MAX,
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

    setattr(TrialConfig, "GATE_ALPHA_LOCAL_WEIGHT", float(params["alpha_local_weight"]))
    setattr(TrialConfig, "GATE_LOCAL_WINDOW_SIZE", int(params["local_window_size"]))
    setattr(TrialConfig, "GATE_GLOBAL_EMA_DECAY", float(params["global_ema_decay"]))
    setattr(TrialConfig, "GATE_THRESHOLD_BIAS", float(params["threshold_bias"]))

    setattr(TrialConfig, "THRESHOLD_QUANTILE", float(params["threshold_quantile"]))
    setattr(TrialConfig, "THRESHOLD_MAD_SCALE", float(params["threshold_mad_scale"]))
    setattr(TrialConfig, "THRESHOLD_LOCAL_UPDATE_RATE", float(params["threshold_local_update_rate"]))
    setattr(TrialConfig, "THRESHOLD_QUANTILE_WEIGHT", float(params["threshold_quantile_weight"]))

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
    logger.info("开始 Gate 多目标超参优化（召回 - 误触发惩罚）")
    logger.info(f"试验次数: {n_trials}, 并行任务数: {n_jobs}")

    recall_weight = float(base_config.GATE_OPT_RECALL_WEIGHT)
    fp_penalty_weight = float(base_config.GATE_OPT_FP_PENALTY_WEIGHT)
    min_action_rate = float(base_config.GATE_OPT_MIN_ACTION_RATE)
    min_action_rate_penalty_weight = float(base_config.GATE_OPT_MIN_ACTION_RATE_PENALTY_WEIGHT)
    max_action_rate = float(base_config.GATE_OPT_MAX_ACTION_RATE)
    max_action_rate_penalty_weight = float(base_config.GATE_OPT_MAX_ACTION_RATE_PENALTY_WEIGHT)
    logger.info(
        "目标函数: composite = "
        f"{recall_weight:.3f}*recall - {fp_penalty_weight:.3f}*false_discovery_rate "
        f"- {min_action_rate_penalty_weight:.3f}*(max(0, {min_action_rate:.4f}-action_rate)/{min_action_rate:.4f})^2 "
        f"- {max_action_rate_penalty_weight:.3f}*(max(0, action_rate-{max_action_rate:.4f})/{max_action_rate:.4f})^2"
    )

    resolved_train_dir = resolve_train_experiment_dir(train_experiment_dir)
    logger.info(f"使用训练实验目录: {resolved_train_dir}")

    test_data, action_space, agent, checkpoint, reward_calc = build_eval_components(
        config=base_config,
        resolved_train_dir=resolved_train_dir,
    )
    prewarm_features = _load_prewarm_features(base_config)

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
            f"alpha={params['alpha_local_weight']:.3f}, q={params['threshold_quantile']:.3f}, "
            f"bias={params['threshold_bias']:.3f}"
        )

        try:
            trial_config = _build_trial_config(base_config, params)

            gate = create_streaming_gate(
                config=trial_config,
                test_data=test_data,
                logger=logger,
                gate_state_path=None,
            )
            for sample in prewarm_features:
                gate.predict(sample)

            env = SequenceEnv(test_data, trial_config.STATE_COLUMNS, reward_calc)
            summary, step_results = evaluate_event_driven(
                agent=agent,
                env=env,
                data=test_data,
                action_space=action_space,
                gate=gate,
                feature_columns=trial_config.FEATURE_COLUMNS,
                supply_temp_ref=trial_config.CHILLER_SUPPLY_TEMP_REF,
            )

            metrics = _calculate_recall_metrics(step_results)
            recall = float(metrics["recall"])
            false_discovery_rate = float(metrics["false_discovery_rate"])
            action_rate = float(metrics["action_rate"])
            action_rate_shortfall = max(0.0, min_action_rate - action_rate)
            action_rate_shortfall_ratio = action_rate_shortfall / max(min_action_rate, 1e-8)
            min_action_penalty = min_action_rate_penalty_weight * (action_rate_shortfall_ratio**2)

            action_rate_excess = max(0.0, action_rate - max_action_rate)
            action_rate_excess_ratio = action_rate_excess / max(max_action_rate, 1e-8)
            max_action_penalty = max_action_rate_penalty_weight * (action_rate_excess_ratio**2)

            composite_score = (
                recall_weight * recall
                - fp_penalty_weight * false_discovery_rate
                - min_action_penalty
                - max_action_penalty
            )

            trial.set_user_attr("recall", recall)
            trial.set_user_attr("precision", float(metrics["precision"]))
            trial.set_user_attr("false_discovery_rate", false_discovery_rate)
            trial.set_user_attr("false_positive_rate", float(metrics["false_positive_rate"]))
            trial.set_user_attr("action_rate", action_rate)
            trial.set_user_attr("action_rate_shortfall", action_rate_shortfall)
            trial.set_user_attr("action_rate_shortfall_ratio", action_rate_shortfall_ratio)
            trial.set_user_attr("min_action_penalty", min_action_penalty)
            trial.set_user_attr("action_rate_excess", action_rate_excess)
            trial.set_user_attr("action_rate_excess_ratio", action_rate_excess_ratio)
            trial.set_user_attr("max_action_penalty", max_action_penalty)
            trial.set_user_attr("total_reward", float(summary["total_reward"]))
            trial.set_user_attr("violation_time_pct", float(summary["violation_time_pct"]))
            trial.set_user_attr("composite_score", composite_score)

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Trial {trial.number} 完成: composite={composite_score:.6f}, recall={recall:.6f}, "
                f"fdr={false_discovery_rate:.6f}, precision={metrics['precision']:.6f}, "
                f"action_rate={action_rate:.4f}, min_action_penalty={min_action_penalty:.6f}, "
                f"max_action_penalty={max_action_penalty:.6f}, "
                f"total_reward={summary['total_reward']:.2f}, "
                f"elapsed={elapsed:.1f}s"
            )

            return -composite_score

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

    best_composite_score = -float(result["best_value"])
    best_params = result["best_params"]
    best_trial = optimizer.study.best_trial if optimizer.study is not None else None
    best_recall = float(best_trial.user_attrs.get("recall", 0.0)) if best_trial is not None else 0.0
    best_precision = float(best_trial.user_attrs.get("precision", 0.0)) if best_trial is not None else 0.0
    best_fdr = float(best_trial.user_attrs.get("false_discovery_rate", 0.0)) if best_trial is not None else 0.0
    best_config_overrides = {
        "GATE_ALPHA_LOCAL_WEIGHT": float(best_params["alpha_local_weight"]),
        "GATE_LOCAL_WINDOW_SIZE": int(best_params["local_window_size"]),
        "GATE_GLOBAL_EMA_DECAY": float(best_params["global_ema_decay"]),
        "GATE_THRESHOLD_BIAS": float(best_params["threshold_bias"]),
        "THRESHOLD_QUANTILE": float(best_params["threshold_quantile"]),
        "THRESHOLD_MAD_SCALE": float(best_params["threshold_mad_scale"]),
        "THRESHOLD_LOCAL_UPDATE_RATE": float(best_params["threshold_local_update_rate"]),
        "THRESHOLD_QUANTILE_WEIGHT": float(best_params["threshold_quantile_weight"]),
        "GATE_SCORE_SHORT_WEIGHT": float(best_params["score_short_weight"]),
        "GATE_SCORE_MEDIUM_WEIGHT": float(best_params["score_medium_weight"]),
        "GATE_SCORE_LONG_WEIGHT": float(
            max(0.01, 1.0 - float(best_params["score_short_weight"]) - float(best_params["score_medium_weight"]))
        ),
        "GATE_TRIGGER_HYSTERESIS_MARGIN": float(best_params["trigger_hysteresis_margin"]),
        "GATE_MIN_TRIGGER_INTERVAL_STEPS": int(best_params["min_trigger_interval_steps"]),
    }

    summary = {
        "train_experiment_dir": str(resolved_train_dir),
        "base_best_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "n_trials": int(result["n_trials"]),
        "objective": {
            "type": "recall_minus_false_discovery_with_action_rate_band_penalty",
            "recall_weight": recall_weight,
            "fp_penalty_weight": fp_penalty_weight,
            "min_action_rate": min_action_rate,
            "min_action_rate_penalty_weight": min_action_rate_penalty_weight,
            "max_action_rate": max_action_rate,
            "max_action_rate_penalty_weight": max_action_rate_penalty_weight,
        },
        "best_composite_score": best_composite_score,
        "best_recall": best_recall,
        "best_precision": best_precision,
        "best_false_discovery_rate": best_fdr,
        "best_params": best_params,
        "best_config_overrides": best_config_overrides,
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
                    "composite_score": trial.user_attrs.get("composite_score"),
                    "recall": trial.user_attrs.get("recall"),
                    "precision": trial.user_attrs.get("precision"),
                    "false_discovery_rate": trial.user_attrs.get("false_discovery_rate"),
                    "false_positive_rate": trial.user_attrs.get("false_positive_rate"),
                    "action_rate": trial.user_attrs.get("action_rate"),
                    "action_rate_shortfall": trial.user_attrs.get("action_rate_shortfall"),
                    "action_rate_shortfall_ratio": trial.user_attrs.get("action_rate_shortfall_ratio"),
                    "min_action_penalty": trial.user_attrs.get("min_action_penalty"),
                    "action_rate_excess": trial.user_attrs.get("action_rate_excess"),
                    "action_rate_excess_ratio": trial.user_attrs.get("action_rate_excess_ratio"),
                    "max_action_penalty": trial.user_attrs.get("max_action_penalty"),
                    "total_reward": trial.user_attrs.get("total_reward"),
                    "violation_time_pct": trial.user_attrs.get("violation_time_pct"),
                    **trial.params,
                }
            )
        pd.DataFrame(records).to_csv(output_dir / "gate_trials.csv", index=False)

    logger.info(
        f"优化完成，最优 composite={best_composite_score:.6f}, recall={best_recall:.6f}, "
        f"precision={best_precision:.6f}, fdr={best_fdr:.6f}"
    )
    logger.info(f"最优配置已保存: {summary_path}")

    return summary
