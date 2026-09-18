"""门控超参多阶段优化（贝叶斯优化 + 验证集时间切分 + holdout 复评）。

入口 optimize_gate_hyperparameters。搜索空间/阶段管理见 gate_optimize_space，
指标与数据切分见 gate_optimize_metrics；本文件保留优化主流程、目标函数与结果落盘。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from et_prl.environments import SequenceEnv
from et_prl.utils import BayesianOptimizer, Logger

from et_prl.evaluation.control.common import (
    build_test_components,
    create_streaming_gate,
    resolve_train_experiment_dir,
)
from et_prl.evaluation.control.strategies.event_driven import run_event_driven
from et_prl.evaluation.control.strategies.fixed_interval import run_fixed_interval
from et_prl.config.loader import load_config

from et_prl.training.gate_optimize_metrics import (
    _calculate_recall_metrics,
    _load_prewarm_features,
    _positive_part,
    _split_validation_for_optimization,
)
from et_prl.training.gate_optimize_space import (
    _build_trial_config,
    _create_search_space,
    _get_phase_params,
    _load_previous_phase_best,
)


class _ObjectiveEvaluator:
    """单个 trial 的目标函数：构建门控、事件驱动评估、计算惩罚项。

    封装一次优化运行的固定上下文（fit 数据、基线、容忍度等），
    使 objective 闭包可测试、可复用。
    """

    def __init__(
        self,
        *,
        base_config,
        fit_data: pd.DataFrame,
        prewarm_features: np.ndarray,
        agent,
        action_space: np.ndarray,
        reward_calc,
        logger: Logger,
        baseline_total_reward: float,
        baseline_energy_daily: float,
        reward_drop_tolerance_ratio: float,
        energy_increase_tolerance_ratio: float,
        reward_drop_penalty_weight: float,
        energy_increase_penalty_weight: float,
    ) -> None:
        self._base_config = base_config
        self._fit_data = fit_data
        self._prewarm_features = prewarm_features
        self._agent = agent
        self._action_space = action_space
        self._reward_calc = reward_calc
        self._logger = logger
        self._baseline_total_reward = baseline_total_reward
        self._baseline_energy_daily = baseline_energy_daily
        self._reward_drop_tolerance_ratio = reward_drop_tolerance_ratio
        self._energy_increase_tolerance_ratio = energy_increase_tolerance_ratio
        self._reward_drop_penalty_weight = reward_drop_penalty_weight
        self._energy_increase_penalty_weight = energy_increase_penalty_weight
        self._active_trials = {"count": 0}
        self._active_lock = threading.Lock()

    def _log_param_summary(self, params: dict) -> str:
        log_params = []
        if "alpha_local_weight" in params:
            log_params.append(f"alpha={params['alpha_local_weight']:.3f}")
        if "global_ema_decay" in params:
            log_params.append(f"ema={params['global_ema_decay']:.4f}")
        if "threshold_quantile" in params:
            log_params.append(f"q={params['threshold_quantile']:.3f}")
        if "threshold_quantile_weight" in params:
            log_params.append(f"qw={params['threshold_quantile_weight']:.3f}")
        if "threshold_bias" in params:
            log_params.append(f"bias={params['threshold_bias']:.3f}")
        if "reference_samples" in params:
            log_params.append(f"ref={params['reference_samples']}")
        if "threshold_mad_scale" in params:
            log_params.append(f"mad={params['threshold_mad_scale']:.3f}")
        if "threshold_local_update_rate" in params:
            log_params.append(f"lr={params['threshold_local_update_rate']:.3f}")
        if "local_window_size" in params:
            log_params.append(f"win={params['local_window_size']}")
        if "score_short_weight" in params:
            log_params.append(f"w_short={params['score_short_weight']:.3f}")
        if "score_medium_weight" in params:
            log_params.append(f"w_med={params['score_medium_weight']:.3f}")
        return ", ".join(log_params)

    def __call__(self, trial, params: dict) -> float:
        start_time = time.perf_counter()
        worker_name = threading.current_thread().name
        with self._active_lock:
            self._active_trials["count"] += 1
            current_active = self._active_trials["count"]

        self._logger.info(
            f"Trial {trial.number} 开始 | worker={worker_name} | active_trials={current_active} | "
            f"{self._log_param_summary(params)}"
        )

        try:
            trial_config = _build_trial_config(self._base_config, params)

            gate = create_streaming_gate(
                config=trial_config,
                test_data=self._fit_data,
                logger=self._logger,
                gate_state_path=None,
            )
            for sample in self._prewarm_features:
                gate.predict(sample)

            env = SequenceEnv(self._fit_data, trial_config.STATE_COLUMNS, self._reward_calc)
            summary, step_results = run_event_driven(
                agent=self._agent,
                env=env,
                data=self._fit_data,
                action_space=self._action_space,
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

            reward_drop_ratio = _positive_part((self._baseline_total_reward - total_reward) / max(abs(self._baseline_total_reward), 1e-8))
            energy_increase_ratio = _positive_part((energy_daily - self._baseline_energy_daily) / max(abs(self._baseline_energy_daily), 1e-8))

            reward_violation_ratio = _positive_part(reward_drop_ratio - self._reward_drop_tolerance_ratio) / max(
                self._reward_drop_tolerance_ratio, 1e-8
            )
            energy_violation_ratio = _positive_part(
                energy_increase_ratio - self._energy_increase_tolerance_ratio
            ) / max(self._energy_increase_tolerance_ratio, 1e-8)

            reward_penalty = self._reward_drop_penalty_weight * (reward_violation_ratio**2)
            energy_penalty = self._energy_increase_penalty_weight * (energy_violation_ratio**2)

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
            self._logger.info(
                f"Trial {trial.number} 完成: objective={objective_score:.6f}, recall={recall:.6f}, "
                f"fdr={false_discovery_rate:.6f}, precision={metrics['precision']:.6f}, "
                f"action_rate={action_rate:.4f}, reward_penalty={reward_penalty:.6f}, "
                f"energy_penalty={energy_penalty:.6f}, "
                f"total_reward={total_reward:.2f}, "
                f"elapsed={elapsed:.1f}s"
            )

            return objective_score

        except Exception:
            elapsed = time.perf_counter() - start_time
            self._logger.exception(f"Trial {trial.number} 失败")
            self._logger.info(f"Trial {trial.number} 结束(失败) | worker={worker_name} | elapsed={elapsed:.1f}s")
            return 1e9

        finally:
            with self._active_lock:
                self._active_trials["count"] -= 1
                current_active = self._active_trials["count"]
            self._logger.info(f"Trial {trial.number} 释放 worker={worker_name} | active_trials={current_active}")


def _merge_best_params(
    previous_best_params: dict | None,
    best_params: dict,
    base_config,
) -> dict:
    """合并多阶段参数：前阶段 best + 当前阶段 best + 配置默认值（min/max 中点）。"""
    merged_best_params: dict = {}
    all_params = [
        "global_ema_decay", "alpha_local_weight", "local_window_size", "reference_samples",
        "threshold_bias", "threshold_quantile", "threshold_mad_scale",
        "threshold_local_update_rate", "threshold_quantile_weight", "threshold_min_samples_for_optimization",
        "score_short_weight", "score_medium_weight", "trigger_hysteresis_margin",
    ]

    for param in all_params:
        if previous_best_params is not None and param in previous_best_params:
            merged_best_params[param] = previous_best_params[param]
        elif param in best_params:
            merged_best_params[param] = best_params[param]
        else:
            min_attr = f"GATE_OPT_{param.upper()}_MIN"
            max_attr = f"GATE_OPT_{param.upper()}_MAX"
            min_val = getattr(base_config, min_attr, None)
            max_val = getattr(base_config, max_attr, None)
            if min_val is not None and max_val is not None:
                default_val = (min_val + max_val) / 2
                merged_best_params[param] = default_val

    return merged_best_params


def _evaluate_holdout(
    *,
    holdout_data: pd.DataFrame,
    best_trial_config,
    agent,
    action_space: np.ndarray,
    reward_calc,
    prewarm_features: np.ndarray,
    baseline_fixed_interval: int,
    base_config,
    reward_drop_tolerance_ratio: float,
    energy_increase_tolerance_ratio: float,
    reward_drop_penalty_weight: float,
    energy_increase_penalty_weight: float,
    best_objective_score: float,
    best_action_rate: float,
    best_recall: float,
    best_precision: float,
    best_fdr: float,
    logger: Logger,
) -> dict:
    """在 holdout 段上复评最优配置，计算泛化差距。"""
    holdout_baseline_env = SequenceEnv(holdout_data, base_config.STATE_COLUMNS, reward_calc)
    holdout_baseline_summary, _ = run_fixed_interval(
        agent=agent,
        env=holdout_baseline_env,
        action_space=action_space,
        fixed_interval=baseline_fixed_interval,
        supply_temp_ref=base_config.CHILLER_SUPPLY_TEMP_REF,
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
    holdout_summary, holdout_step_results = run_event_driven(
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

    holdout_test_result = {
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

    return holdout_test_result


def optimize_gate_hyperparameters(
    train_experiment_dir: Path | None = None,
    n_trials: int | None = None,
    n_jobs: int | None = None,
    previous_phase_result_dir: Path | None = None,
) -> dict:
    base_config = load_config("control_compare")
    # 多阶段优化配置
    current_phase = getattr(base_config, "GATE_OPTIMIZATION_PHASE", "phase1")
    phase_cfg = _get_phase_params(current_phase)
    phase_shrink_ratio = float(getattr(base_config, "GATE_OPTIMIZATION_PHASE_RANGE_SHRINK_RATIO", 0.2))

    if n_trials is None:
        # 根据phase使用对应的trials数
        phase_trials_attr = f"GATE_OPTIMIZATION_{current_phase.upper()}_TRIALS"
        n_trials = int(getattr(base_config, phase_trials_attr, base_config.GATE_OPTIMIZATION_DEFAULT_TRIALS))
    if n_jobs is None:
        n_jobs = int(base_config.GATE_OPTIMIZATION_DEFAULT_N_JOBS)

    # 为每个phase创建独立目录
    base_output_dir = base_config.get_optimization_dir() / base_config.TIMESTAMP
    output_dir = base_output_dir / current_phase / "gate"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(output_dir, log_filename="run.log")

    # 设置清晰的日志头
    logger.info(f"多阶段超参优化调度")
    logger.info(f"当前阶段: {current_phase.upper()}")
    logger.info(f"阶段描述: {phase_cfg['description']}")
    logger.info(f"试验次数: {n_trials}, 并行任务数: {n_jobs}")
    logger.info(f"输出目录: {output_dir}")

    # 加载前一阶段的最优参数（仅phase2/3）
    previous_best_params = _load_previous_phase_best(
        output_dir, current_phase, previous_phase_result_dir, logger
    )

    if previous_best_params is not None:
        logger.info(f"搜索范围: 在前阶段最优值周围±{phase_shrink_ratio:.0%}缩小")

    logger.info(f"本阶段优化参数({len(phase_cfg['params'])}个): {phase_cfg['params']}")

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

    val_data, action_space, agent, checkpoint, reward_calc = build_test_components(
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
    baseline_summary, _ = run_fixed_interval(
        agent=agent,
        env=baseline_env,
        action_space=action_space,
        fixed_interval=baseline_fixed_interval,
        supply_temp_ref=base_config.CHILLER_SUPPLY_TEMP_REF,
    )
    baseline_total_reward = float(baseline_summary["total_reward"])
    baseline_energy_daily = float(baseline_summary["E_daily_kwh_per_day"])
    baseline_action_rate = float(baseline_summary["action_frequency"])
    logger.info(
        "验证集基线(fixed_interval="
        f"{baseline_fixed_interval}): reward={baseline_total_reward:.2f}, "
        f"energy_daily={baseline_energy_daily:.4f}, action_rate={baseline_action_rate:.4f}"
    )

    space = _create_search_space(
        config=base_config,
        phase=current_phase,
        previous_best_params=previous_best_params,
        shrink_ratio=phase_shrink_ratio,
    )
    optimizer = BayesianOptimizer(
        space=space,
        output_dir=output_dir,
        results_dir=output_dir / "results",
        sampler=str(base_config.GATE_OPTIMIZATION_SAMPLER),
        seed=int(base_config.GATE_OPTIMIZATION_SEED),
    )

    evaluator = _ObjectiveEvaluator(
        base_config=base_config,
        fit_data=fit_data,
        prewarm_features=prewarm_features,
        agent=agent,
        action_space=action_space,
        reward_calc=reward_calc,
        logger=logger,
        baseline_total_reward=baseline_total_reward,
        baseline_energy_daily=baseline_energy_daily,
        reward_drop_tolerance_ratio=reward_drop_tolerance_ratio,
        energy_increase_tolerance_ratio=energy_increase_tolerance_ratio,
        reward_drop_penalty_weight=reward_drop_penalty_weight,
        energy_increase_penalty_weight=energy_increase_penalty_weight,
    )

    result = optimizer.optimize(
        objective_fn=evaluator,
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

    # 合并多阶段参数：前阶段best + 当前阶段best + 配置默认值
    merged_best_params = _merge_best_params(previous_best_params, best_params, base_config)

    # 保存本阶段best_params为JSON，供下一阶段加载
    best_params_json_path = output_dir / "results" / "best_params.json"
    with open(best_params_json_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存 {current_phase} 最优参数到: {best_params_json_path}")

    best_trial_config = _build_trial_config(base_config, merged_best_params)

    best_config_overrides = {
        "GATE_GLOBAL_EMA_DECAY": float(merged_best_params["global_ema_decay"]),
        "GATE_ALPHA_LOCAL_WEIGHT": float(merged_best_params["alpha_local_weight"]),
        "GATE_LOCAL_WINDOW_SIZE": int(merged_best_params["local_window_size"]),
        "GATE_REFERENCE_SAMPLES": int(merged_best_params["reference_samples"]),
        "GATE_THRESHOLD_BIAS": float(merged_best_params["threshold_bias"]),
        "THRESHOLD_QUANTILE": float(merged_best_params["threshold_quantile"]),
        "THRESHOLD_MAD_SCALE": float(merged_best_params["threshold_mad_scale"]),
        "THRESHOLD_LOCAL_UPDATE_RATE": float(merged_best_params["threshold_local_update_rate"]),
        "THRESHOLD_QUANTILE_WEIGHT": float(merged_best_params["threshold_quantile_weight"]),
        "THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION": int(merged_best_params["threshold_min_samples_for_optimization"]),
        "GATE_SCORE_SHORT_WEIGHT": float(merged_best_params["score_short_weight"]),
        "GATE_SCORE_MEDIUM_WEIGHT": float(merged_best_params["score_medium_weight"]),
        "GATE_SCORE_LONG_WEIGHT": float(
            max(0.01, 1.0 - float(merged_best_params["score_short_weight"]) - float(merged_best_params["score_medium_weight"]))
        ),
        "GATE_TRIGGER_HYSTERESIS_MARGIN": float(merged_best_params["trigger_hysteresis_margin"]),
    }

    holdout_test_result: dict | None = None
    if holdout_data is not None and not holdout_data.empty:
        holdout_test_result = _evaluate_holdout(
            holdout_data=holdout_data,
            best_trial_config=best_trial_config,
            agent=agent,
            action_space=action_space,
            reward_calc=reward_calc,
            prewarm_features=prewarm_features,
            baseline_fixed_interval=baseline_fixed_interval,
            base_config=base_config,
            reward_drop_tolerance_ratio=reward_drop_tolerance_ratio,
            energy_increase_tolerance_ratio=energy_increase_tolerance_ratio,
            reward_drop_penalty_weight=reward_drop_penalty_weight,
            energy_increase_penalty_weight=energy_increase_penalty_weight,
            best_objective_score=best_objective_score,
            best_action_rate=best_action_rate,
            best_recall=best_recall,
            best_precision=best_precision,
            best_fdr=best_fdr,
            logger=logger,
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
        "holdout_evaluation": holdout_test_result,
    }

    summary_path = output_dir / "results" / "best_gate_config.json"
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
        pd.DataFrame(records).to_csv(output_dir / "results" / "gate_trials.csv", index=False)

    logger.info(
        f"优化完成，最优 objective={best_objective_score:.6f}, action_rate={best_action_rate:.6f}, "
        f"recall={best_recall:.6f}, precision={best_precision:.6f}, fdr={best_fdr:.6f}"
    )
    logger.info(f"最优配置已保存: {summary_path}")

    return summary
