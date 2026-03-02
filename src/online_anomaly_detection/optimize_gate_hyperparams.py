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

    return TrialConfig


def _calculate_recall_metrics(step_results: pd.DataFrame) -> dict:
    violation = (
        (step_results["chiller_supply_temp"] < 15.0) | (step_results["chiller_supply_temp"] > 19.0)
    ).astype(int)
    gate_signal = step_results["gate_signal"].astype(int)

    tp = int(((gate_signal == 1) & (violation == 1)).sum())
    fp = int(((gate_signal == 1) & (violation == 0)).sum())
    fn = int(((gate_signal == 0) & (violation == 1)).sum())

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    action_rate = float(step_results["action_updated"].mean()) if not step_results.empty else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
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

    output_dir = base_config.get_optimization_dir() / base_config.TIMESTAMP / "gate"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(output_dir)
    logger.info("开始 Gate 召回超参优化")
    logger.info(f"试验次数: {n_trials}, 并行任务数: {n_jobs}")

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

            trial.set_user_attr("recall", recall)
            trial.set_user_attr("precision", float(metrics["precision"]))
            trial.set_user_attr("action_rate", float(metrics["action_rate"]))
            trial.set_user_attr("total_reward", float(summary["total_reward"]))
            trial.set_user_attr("violation_time_pct", float(summary["violation_time_pct"]))

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Trial {trial.number} 完成: recall={recall:.6f}, precision={metrics['precision']:.6f}, "
                f"action_rate={metrics['action_rate']:.4f}, total_reward={summary['total_reward']:.2f}, "
                f"elapsed={elapsed:.1f}s"
            )

            return -recall

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

    best_recall = -float(result["best_value"])
    best_params = result["best_params"]
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
    }

    summary = {
        "train_experiment_dir": str(resolved_train_dir),
        "base_best_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "n_trials": int(result["n_trials"]),
        "best_recall": best_recall,
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
                    "recall": trial.user_attrs.get("recall"),
                    "precision": trial.user_attrs.get("precision"),
                    "action_rate": trial.user_attrs.get("action_rate"),
                    "total_reward": trial.user_attrs.get("total_reward"),
                    "violation_time_pct": trial.user_attrs.get("violation_time_pct"),
                    **trial.params,
                }
            )
        pd.DataFrame(records).to_csv(output_dir / "gate_trials.csv", index=False)

    logger.info(f"优化完成，最优召回: {best_recall:.6f}")
    logger.info(f"最优配置已保存: {summary_path}")

    return summary
