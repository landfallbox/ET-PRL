from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from et_prl.environments import SequenceEnv
from et_prl.utils import BayesianOptimizer, HyperparameterSpace, Logger

from et_prl.config.control_compare import ControlCompareConfig
from et_prl.config.gate import OnlineAnomalyDetectionConfig
from et_prl.evaluation.control.common import (
    build_test_components,
    create_streaming_gate,
    resolve_train_experiment_dir,
)
from et_prl.evaluation.control.strategies.event_driven import test_event_driven
from et_prl.evaluation.control.strategies.fixed_interval import test_fixed_interval


def _load_prewarm_features(
    config: type[ControlCompareConfig],
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


def _get_dynamic_param_range(
    param_name: str,
    config: type[ControlCompareConfig],
    phase: str,
    previous_best_value: float | None,
    shrink_ratio: float = 0.2,
) -> tuple[float | int, float | int]:
    """
    根据阶段和前阶段最优值，计算动态搜索范围
    
    参数：
        param_name: 参数名（如'global_ema_decay'）
        config: 配置类
        phase: 当前阶段（'phase1', 'phase2', 'phase3'）
        previous_best_value: 前阶段的最优值（None表示phase1或不需要动态缩小）
        shrink_ratio: 范围缩小比例（默认±20%）
    """
    min_attr = f"GATE_OPT_{param_name.upper()}_MIN"
    max_attr = f"GATE_OPT_{param_name.upper()}_MAX"
    
    default_min = getattr(config, min_attr)
    default_max = getattr(config, max_attr)
    
    # Phase 1: 使用原始范围
    if phase == "phase1" or previous_best_value is None:
        return default_min, default_max
    
    # Phase 2/3: 在前阶段最优值周围缩小范围
    range_width = default_max - default_min
    shrink_amount = range_width * shrink_ratio
    
    new_min = max(default_min, previous_best_value - shrink_amount)
    new_max = min(default_max, previous_best_value + shrink_amount)
    
    return new_min, new_max


def _get_phase_params(phase: str) -> dict:
    """获取每个阶段应该优化的参数列表"""
    params_config = {
        "phase1": {
            "description": "Layer A: 核心决策参数（5个）",
            "params": [
                "threshold_bias",
                "trigger_hysteresis_margin",
                "threshold_quantile",
                "threshold_mad_scale",
                "threshold_local_update_rate",
            ],
            "trials": 60,
        },
        "phase2": {
            "description": "Layer B: 自适应参数（5个）",
            "params": [
                "alpha_local_weight",
                "global_ema_decay",
                "threshold_min_samples_for_optimization",
                "local_window_size",
                "reference_samples",
            ],
            "trials": 50,
        },
        "phase3": {
            "description": "Layer C+D: 融合与风格参数（3个）",
            "params": [
                "score_short_weight",
                "score_medium_weight",
                "threshold_quantile_weight",
            ],
            "trials": 40,
        },
    }
    return params_config.get(phase, params_config["phase1"])


def _load_previous_phase_best(
    output_dir: Path,
    current_phase: str,
    previous_phase_result_dir: Path | None = None,
    logger: Logger | None = None,
) -> dict | None:
    """从前一个阶段加载最优参数
    
    Args:
        output_dir: 当前phase的输出目录
        current_phase: 当前阶段（'phase1', 'phase2', 'phase3'）
        previous_phase_result_dir: 显式指定的前阶段结果目录路径
        logger: 日志对象，用于详细的输出信息
    
    Returns:
        前阶段最优参数dict，或None（phase1或找不到结果）
    """
    if current_phase == "phase1":
        if logger:
            logger.info("[Phase 1] 初始阶段，无前阶段结果可继承")
        return None
    
    phase_order = {"phase2": "phase1", "phase3": "phase2"}
    prev_phase = phase_order.get(current_phase)
    if prev_phase is None:
        return None
    
    # 确定前阶段最优参数文件的路径
    candidate_paths: list[Path] = []
    if previous_phase_result_dir is not None:
        # 显式路径兼容两种输入：
        # 1) .../phase1
        # 2) .../phase1/gate
        candidate_paths = [
            previous_phase_result_dir / "best_params.json",
            previous_phase_result_dir / "gate" / "best_params.json",
        ]
        source_hint = f"(显式指定: {previous_phase_result_dir})"
    else:
        # 回退：自动检测（相同timestamp）下的标准gate目录
        candidate_paths = [
            output_dir.parent / prev_phase / "gate" / "best_params.json",
            output_dir.parent / prev_phase / "best_params.json",
        ]
        source_hint = f"(自动检测: {output_dir.parent})"

    prev_best_path = next((path for path in candidate_paths if path.exists()), None)

    if prev_best_path is None:
        if logger:
            logger.warning(
                f"[{current_phase.upper()}] 前阶段结果不存在 {source_hint}\n"
                f"      尝试路径: {[str(p) for p in candidate_paths]}\n"
                f"      这可能意味着 {prev_phase} 尚未运行，请先执行该阶段"
            )
        return None
    
    try:
        with open(prev_best_path) as f:
            params = json.load(f)
        if logger:
            logger.info(
                f"[{current_phase.upper()}] ✓ 已加载 {prev_phase.upper()} 的最优参数 {source_hint}"
            )
            logger.info(
                f"             文件: {prev_best_path}"
            )
            logger.info(
                f"             参数: {list(params.keys())}"
            )
        return params
    except Exception as exc:
        if logger:
            logger.error(
                f"[{current_phase.upper()}] 加载前阶段最优参数失败: {prev_best_path} | 异常: {exc}"
            )
        return None


def _create_search_space(
    config: type[ControlCompareConfig],
    phase: str = "phase1",
    previous_best_params: dict | None = None,
    shrink_ratio: float = 0.2,
) -> HyperparameterSpace:
    """
    构建超参搜索空间（支持多阶段）
    
    参数：
        config: 配置类
        phase: 当前优化阶段 ('phase1', 'phase2', 'phase3')
        previous_best_params: 前一阶段的最优参数
        shrink_ratio: 搜索范围缩小比例（相对于原始范围）
    """
    phase_cfg = _get_phase_params(phase)
    params_to_optimize = phase_cfg["params"]
    
    space = HyperparameterSpace()
    
    # 参数->属性名的映射，便于在phase2/3中固定前阶段的参数
    param_mapping = {
        "global_ema_decay": ("float", config.GATE_OPT_GLOBAL_EMA_DECAY_MIN, config.GATE_OPT_GLOBAL_EMA_DECAY_MAX),
        "alpha_local_weight": ("float", config.GATE_OPT_ALPHA_LOCAL_WEIGHT_MIN, config.GATE_OPT_ALPHA_LOCAL_WEIGHT_MAX),
        "local_window_size": ("int", config.GATE_OPT_LOCAL_WINDOW_SIZE_MIN, config.GATE_OPT_LOCAL_WINDOW_SIZE_MAX),
        "reference_samples": ("int", config.GATE_OPT_REFERENCE_SAMPLES_MIN, config.GATE_OPT_REFERENCE_SAMPLES_MAX),
        "threshold_bias": ("float", config.GATE_OPT_THRESHOLD_BIAS_MIN, config.GATE_OPT_THRESHOLD_BIAS_MAX),
        "threshold_quantile": ("float", config.GATE_OPT_THRESHOLD_QUANTILE_MIN, config.GATE_OPT_THRESHOLD_QUANTILE_MAX),
        "threshold_mad_scale": ("float", config.GATE_OPT_THRESHOLD_MAD_SCALE_MIN, config.GATE_OPT_THRESHOLD_MAD_SCALE_MAX),
        "threshold_local_update_rate": ("float", config.GATE_OPT_THRESHOLD_LOCAL_UPDATE_RATE_MIN, config.GATE_OPT_THRESHOLD_LOCAL_UPDATE_RATE_MAX),
        "threshold_quantile_weight": ("float", config.GATE_OPT_THRESHOLD_QUANTILE_WEIGHT_MIN, config.GATE_OPT_THRESHOLD_QUANTILE_WEIGHT_MAX),
        "threshold_min_samples_for_optimization": ("int", config.GATE_OPT_THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION_MIN, config.GATE_OPT_THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION_MAX),
        "score_short_weight": ("float", config.GATE_OPT_SCORE_SHORT_WEIGHT_MIN, config.GATE_OPT_SCORE_SHORT_WEIGHT_MAX),
        "score_medium_weight": ("float", config.GATE_OPT_SCORE_MEDIUM_WEIGHT_MIN, config.GATE_OPT_SCORE_MEDIUM_WEIGHT_MAX),
        "trigger_hysteresis_margin": ("float", config.GATE_OPT_TRIGGER_HYSTERESIS_MARGIN_MIN, config.GATE_OPT_TRIGGER_HYSTERESIS_MARGIN_MAX),
    }
    
    # 按顺序添加参数
    for idx, param_name in enumerate(params_to_optimize):
        if param_name not in param_mapping:
            continue
        
        param_type, default_min, default_max = param_mapping[param_name]
        
        # 如果不是phase1，尝试从previous_best_params获取最优值并缩小范围
        prev_value = None
        if previous_best_params is not None and param_name in previous_best_params:
            prev_value = previous_best_params[param_name]
        
        min_val, max_val = _get_dynamic_param_range(
            param_name,
            config,
            phase,
            prev_value,
            shrink_ratio,
        )
        
        if param_type == "float":
            space.add_float(param_name, min_val, max_val)
        else:  # int
            space.add_int(param_name, int(min_val), int(max_val))
    
    return space


def _build_trial_config(base_cls: type[ControlCompareConfig], params: dict) -> type[ControlCompareConfig]:
    class TrialConfig(base_cls):
        pass

    # 参数映射：参数名 -> (config属性名, 类型)
    param_config_mapping = {
        "global_ema_decay": ("GATE_GLOBAL_EMA_DECAY", float),
        "alpha_local_weight": ("GATE_ALPHA_LOCAL_WEIGHT", float),
        "local_window_size": ("GATE_LOCAL_WINDOW_SIZE", int),
        "reference_samples": ("GATE_REFERENCE_SAMPLES", int),
        "threshold_bias": ("GATE_THRESHOLD_BIAS", float),
        "threshold_quantile": ("THRESHOLD_QUANTILE", float),
        "threshold_mad_scale": ("THRESHOLD_MAD_SCALE", float),
        "threshold_local_update_rate": ("THRESHOLD_LOCAL_UPDATE_RATE", float),
        "threshold_quantile_weight": ("THRESHOLD_QUANTILE_WEIGHT", float),
        "threshold_min_samples_for_optimization": ("THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION", int),
        "score_short_weight": ("GATE_SCORE_SHORT_WEIGHT", float),
        "score_medium_weight": ("GATE_SCORE_MEDIUM_WEIGHT", float),
        "trigger_hysteresis_margin": ("GATE_TRIGGER_HYSTERESIS_MARGIN", float),
    }
    
    # 针对score_weight需要特别处理
    has_score_short = "score_short_weight" in params
    has_score_medium = "score_medium_weight" in params
    
    for param_name, (config_attr, param_type) in param_config_mapping.items():
        if param_name in params:
            # 使用优化进来的值
            value = param_type(params[param_name])
        else:
            # 使用base_config的默认值
            value = param_type(getattr(base_cls, config_attr))
        
        setattr(TrialConfig, config_attr, value)
    
    # 计算score_long_weight
    if has_score_short and has_score_medium:
        score_short_weight = float(params["score_short_weight"])
        score_medium_weight = float(params["score_medium_weight"])
    else:
        score_short_weight = float(getattr(base_cls, "GATE_SCORE_SHORT_WEIGHT"))
        score_medium_weight = float(getattr(base_cls, "GATE_SCORE_MEDIUM_WEIGHT"))
    
    score_long_weight = max(0.01, 1.0 - score_short_weight - score_medium_weight)
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
    previous_phase_result_dir: Path | None = None,
) -> dict:
    base_config = ControlCompareConfig
    
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
    base_output_dir = OnlineAnomalyDetectionConfig.get_optimization_dir() / OnlineAnomalyDetectionConfig.TIMESTAMP
    output_dir = base_output_dir / current_phase / "gate"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(output_dir)
    
    # 设置清晰的日志头
    logger.info("" + "="*70)
    logger.info(f"多阶段超参优化调度")
    logger.info("="*70)
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
    logger.info("")

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
    baseline_summary, _ = test_fixed_interval(
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

    space = _create_search_space(
        config=base_config,
        phase=current_phase,
        previous_best_params=previous_best_params,
        shrink_ratio=phase_shrink_ratio,
    )
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

        # 构建日志字符串，只记录存在的参数
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
        
        logger.info(
            f"Trial {trial.number} 开始 | worker={worker_name} | active_trials={current_active} | "
            f"{', '.join(log_params)}"
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
            summary, step_results = test_event_driven(
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

        except Exception:
            elapsed = time.perf_counter() - start_time
            logger.exception(f"Trial {trial.number} 失败")
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
    
    # 合并多阶段参数：前阶段best + 当前阶段best + 配置默认值
    merged_best_params = {}
    all_params = [
        "global_ema_decay", "alpha_local_weight", "local_window_size", "reference_samples",
        "threshold_bias", "threshold_quantile", "threshold_mad_scale",
        "threshold_local_update_rate", "threshold_quantile_weight", "threshold_min_samples_for_optimization",
        "score_short_weight", "score_medium_weight", "trigger_hysteresis_margin",
    ]
    
    # 初始化：使用config默认值或previous_best_params
    for param in all_params:
        if previous_best_params is not None and param in previous_best_params:
            merged_best_params[param] = previous_best_params[param]
        elif param in best_params:
            merged_best_params[param] = best_params[param]
        else:
            # 从config获取默认值
            min_attr = f"GATE_OPT_{param.upper()}_MIN"
            max_attr = f"GATE_OPT_{param.upper()}_MAX"
            min_val = getattr(base_config, min_attr, None)
            max_val = getattr(base_config, max_attr, None)
            if min_val is not None and max_val is not None:
                default_val = (min_val + max_val) / 2
                merged_best_params[param] = default_val
    
    # 保存本阶段best_params为JSON，供下一阶段加载
    best_params_json_path = output_dir / "best_params.json"
    with open(best_params_json_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存 {current_phase} 最优参数到: {best_params_json_path}")
    
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
        best_trial_config = _build_trial_config(base_config, merged_best_params)

        holdout_baseline_env = SequenceEnv(holdout_data, base_config.STATE_COLUMNS, reward_calc)
        holdout_baseline_summary, _ = test_fixed_interval(
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
        holdout_summary, holdout_step_results = test_event_driven(
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
