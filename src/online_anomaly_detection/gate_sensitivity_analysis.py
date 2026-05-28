from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ml_toolkit.rl import SequenceEnv
from ml_toolkit.utils import create_experiment_context

from config.control_compare_config import ControlCompareConfig
from src.control_evaluation.common import build_test_components, create_streaming_gate, resolve_train_experiment_dir
from src.control_evaluation.strategies.event_driven import test_event_driven
from src.control_evaluation.strategies.fixed_interval import test_fixed_interval


@dataclass(frozen=True)
class SensitivitySpec:
    name: str
    config_attr: str
    label: str
    values: tuple[Any, ...]
    group: str
    integer: bool = False


DEFAULT_SPECS: tuple[SensitivitySpec, ...] = (
    # ── Layer A: 核心决策参数 ────────────────────────────────
    SensitivitySpec(
        name="threshold_bias",
        config_attr="GATE_THRESHOLD_BIAS",
        label="Threshold Bias",
        values=(-0.25, -0.21, -0.17, -0.13, -0.09, -0.05, -0.01, 0.03, 0.07, 0.10),
        group="threshold",
    ),
    SensitivitySpec(
        name="threshold_quantile",
        config_attr="THRESHOLD_QUANTILE",
        label="Threshold Quantile",
        values=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
        group="threshold",
    ),
    SensitivitySpec(
        name="threshold_mad_scale",
        config_attr="THRESHOLD_MAD_SCALE",
        label="MAD Scale",
        values=(0.80, 1.09, 1.37, 1.66, 1.94, 2.23, 2.51, 2.80),
        group="threshold",
    ),
    # ── Layer B: 自适应参数 ─────────────────────────────────
    SensitivitySpec(
        name="threshold_local_update_rate",
        config_attr="THRESHOLD_LOCAL_UPDATE_RATE",
        label="Local Update Rate",
        values=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
        group="threshold",
    ),
    SensitivitySpec(
        name="trigger_hysteresis_margin",
        config_attr="GATE_TRIGGER_HYSTERESIS_MARGIN",
        label="Hysteresis Margin",
        values=(0.00, 0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.12),
        group="threshold",
    ),
    SensitivitySpec(
        name="alpha_local_weight",
        config_attr="GATE_ALPHA_LOCAL_WEIGHT",
        label="Alpha Local Weight",
        values=(0.40, 0.48, 0.56, 0.63, 0.71, 0.79, 0.87, 0.95),
        group="adaptive",
    ),
    SensitivitySpec(
        name="local_window_size",
        config_attr="GATE_LOCAL_WINDOW_SIZE",
        label="Local Window Size",
        values=(30, 60, 90, 120, 150, 180, 210, 240),
        group="statistical",
        integer=True,
    ),
    SensitivitySpec(
        name="global_ema_decay",
        config_attr="GATE_GLOBAL_EMA_DECAY",
        label="Global EMA Decay",
        values=(0.005, 0.021, 0.038, 0.054, 0.070, 0.086, 0.103, 0.120),
        group="statistical",
    ),
    # ── Layer C: 融合参数 ───────────────────────────────────
    SensitivitySpec(
        name="score_short_weight",
        config_attr="GATE_SCORE_SHORT_WEIGHT",
        label="Short Weight",
        values=(0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66, 0.72),
        group="fusion",
    ),
    SensitivitySpec(
        name="score_medium_weight",
        config_attr="GATE_SCORE_MEDIUM_WEIGHT",
        label="Medium Weight",
        values=(0.05, 0.11, 0.17, 0.22, 0.28, 0.33, 0.39, 0.44),
        group="fusion",
    ),
    SensitivitySpec(
        name="threshold_quantile_weight",
        config_attr="THRESHOLD_QUANTILE_WEIGHT",
        label="Quantile Weight",
        values=(0.20, 0.31, 0.42, 0.54, 0.65, 0.76, 0.87, 0.95),
        group="threshold",
    ),
)

DEFAULT_PARAM_ORDER: tuple[str, ...] = (
    "threshold_bias",
    "threshold_quantile",
    "threshold_mad_scale",
    "threshold_local_update_rate",
    "threshold_quantile_weight",
    "trigger_hysteresis_margin",
    "alpha_local_weight",
    "local_window_size",
    "global_ema_decay",
    "score_short_weight",
    "score_medium_weight",
)
SPEC_BY_NAME: dict[str, SensitivitySpec] = {spec.name: spec for spec in DEFAULT_SPECS}
PREWARM_SPLITS: tuple[str, ...] = ("train", "val")
ZERO_SENSITIVITY_ABS_TOL = 1e-9


class _SilentLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None

    def error(self, *args: Any, **kwargs: Any) -> None:
        return None


def _resolve_parameters(parameters: list[str] | None) -> list[SensitivitySpec]:
    if not parameters:
        return [SPEC_BY_NAME[name] for name in DEFAULT_PARAM_ORDER]

    resolved: list[SensitivitySpec] = []
    for name in parameters:
        normalized = str(name).strip()
        if normalized not in SPEC_BY_NAME:
            raise ValueError(f"Unknown sensitivity parameter: {normalized}")
        if normalized not in [spec.name for spec in resolved]:
            resolved.append(SPEC_BY_NAME[normalized])
    return resolved


def _coerce_value(spec: SensitivitySpec, value: Any) -> Any:
    if spec.integer:
        return int(round(float(value)))
    return float(value)


def _build_analysis_config(base_cls: type[Any], overrides: dict[str, Any]) -> type[Any]:
    class AnalysisConfig(base_cls):
        pass

    for key, value in overrides.items():
        setattr(AnalysisConfig, key, value)

    return AnalysisConfig


def _build_candidate_overrides(base_config: type[Any], spec: SensitivitySpec, candidate_value: Any) -> dict[str, Any]:
    candidate_value = _coerce_value(spec, candidate_value)

    if spec.name == "score_short_weight":
        medium_weight = float(getattr(base_config, "GATE_SCORE_MEDIUM_WEIGHT"))
        long_weight = 1.0 - float(candidate_value) - medium_weight
        if long_weight <= 0:
            raise ValueError(
                f"Invalid score weights for {spec.name}: short={candidate_value}, medium={medium_weight}, long={long_weight}"
            )
        return {
            "GATE_SCORE_SHORT_WEIGHT": float(candidate_value),
            "GATE_SCORE_MEDIUM_WEIGHT": medium_weight,
            "GATE_SCORE_LONG_WEIGHT": float(long_weight),
        }

    if spec.name == "score_medium_weight":
        short_weight = float(getattr(base_config, "GATE_SCORE_SHORT_WEIGHT"))
        long_weight = 1.0 - short_weight - float(candidate_value)
        if long_weight <= 0:
            raise ValueError(
                f"Invalid score weights for {spec.name}: short={short_weight}, medium={candidate_value}, long={long_weight}"
            )
        return {
            "GATE_SCORE_SHORT_WEIGHT": short_weight,
            "GATE_SCORE_MEDIUM_WEIGHT": float(candidate_value),
            "GATE_SCORE_LONG_WEIGHT": float(long_weight),
        }

    return {spec.config_attr: candidate_value}


def _load_prewarm_features(config: type[Any]) -> np.ndarray:
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


def _prewarm_gate(gate: Any, prewarm_features: np.ndarray) -> None:
    for sample in prewarm_features:
        gate.predict(sample)


def _metric_summary(
    summary: dict,
    step_results: pd.DataFrame,
    default_baseline_summary: dict,
    fixed_baseline_summary: dict,
) -> dict[str, float | int]:
    total_reward = float(summary.get("total_reward", 0.0))
    energy_daily = float(summary.get("E_daily_kwh_per_day", 0.0))
    default_reward = float(default_baseline_summary.get("total_reward", 0.0))
    default_energy_daily = float(default_baseline_summary.get("E_daily_kwh_per_day", 0.0))
    default_action_frequency = float(default_baseline_summary.get("action_frequency", 0.0))
    default_event_trigger_rate = float(default_baseline_summary.get("event_trigger_rate", 0.0))

    baseline_reward = float(fixed_baseline_summary.get("total_reward", 0.0))
    baseline_energy_daily = float(fixed_baseline_summary.get("E_daily_kwh_per_day", 0.0))

    signed_reward_gap_ratio = (baseline_reward - total_reward) / max(abs(baseline_reward), 1e-8)
    signed_energy_change_ratio = (energy_daily - baseline_energy_daily) / max(abs(baseline_energy_daily), 1e-8)

    event_trigger_rate = float(step_results["gate_signal"].eq(1).mean()) if not step_results.empty else 0.0

    return {
        "total_reward": total_reward,
        "avg_reward_per_action": float(summary.get("avg_reward_per_action", 0.0)),
        "avg_reward_per_env_step": float(summary.get("avg_reward_per_env_step", 0.0)),
        "avg_energy_score": float(summary.get("avg_energy_score", 0.0)),
        "avg_comfort_score": float(summary.get("avg_comfort_score", 0.0)),
        "action_count": int(summary.get("action_count", 0)),
        "action_frequency": float(summary.get("action_frequency", 0.0)),
        "event_trigger_count": int(summary.get("event_trigger_count", 0)),
        "event_trigger_rate": float(summary.get("event_trigger_rate", event_trigger_rate)),
        "E_daily_kwh_per_day": energy_daily,
        "delta_total_reward_vs_default_gate": float(total_reward - default_reward),
        "delta_action_frequency_vs_default_gate": float(float(summary.get("action_frequency", 0.0)) - default_action_frequency),
        "delta_event_trigger_rate_vs_default_gate": float(float(summary.get("event_trigger_rate", event_trigger_rate)) - default_event_trigger_rate),
        "delta_E_daily_kwh_per_day_vs_default_gate": float(energy_daily - default_energy_daily),
        "signed_reward_gap_ratio_vs_fixed_baseline": float(signed_reward_gap_ratio),
        "signed_energy_change_ratio_vs_fixed_baseline": float(signed_energy_change_ratio),
    }


def _series_from_step_results(step_results: pd.DataFrame) -> dict[str, np.ndarray]:
    steps = len(step_results)
    if steps == 0:
        return {
            "reward": np.asarray([], dtype=np.float64),
            "action_updated": np.asarray([], dtype=np.float64),
            "gate_signal": np.asarray([], dtype=np.float64),
            "daily_energy": np.asarray([], dtype=np.float64),
        }

    dt_hours = 5.0 / 60.0
    duration_days = max(float(steps) * dt_hours / 24.0, 1e-12)
    return {
        "reward": step_results["reward"].to_numpy(dtype=np.float64),
        "action_updated": step_results["action_updated"].to_numpy(dtype=np.float64),
        "gate_signal": step_results["gate_signal"].to_numpy(dtype=np.float64),
        "daily_energy": step_results["power_chiller"].to_numpy(dtype=np.float64) * dt_hours / duration_days,
    }


def _paired_metric_deltas(candidate_step_results: pd.DataFrame, baseline_step_results: pd.DataFrame) -> dict[str, np.ndarray]:
    candidate_series = _series_from_step_results(candidate_step_results)
    baseline_series = _series_from_step_results(baseline_step_results)
    common_length = min(len(next(iter(candidate_series.values()))), len(next(iter(baseline_series.values()))))
    if common_length <= 0:
        return {
            "total_reward": np.asarray([], dtype=np.float64),
            "avg_reward_per_env_step": np.asarray([], dtype=np.float64),
            "action_frequency": np.asarray([], dtype=np.float64),
            "event_trigger_rate": np.asarray([], dtype=np.float64),
            "E_daily_kwh_per_day": np.asarray([], dtype=np.float64),
        }

    reward_delta = candidate_series["reward"][:common_length] - baseline_series["reward"][:common_length]
    return {
        "total_reward": reward_delta,
        "avg_reward_per_env_step": reward_delta,
        "action_frequency": candidate_series["action_updated"][:common_length] - baseline_series["action_updated"][:common_length],
        "event_trigger_rate": candidate_series["gate_signal"][:common_length] - baseline_series["gate_signal"][:common_length],
        "E_daily_kwh_per_day": candidate_series["daily_energy"][:common_length] - baseline_series["daily_energy"][:common_length],
    }


def _bootstrap_statistic(diff: np.ndarray, metric_name: str, indices: np.ndarray) -> float:
    if diff.size == 0 or indices.size == 0:
        return 0.0
    if metric_name in {"avg_reward_per_env_step", "action_frequency", "event_trigger_rate"}:
        return float(np.mean(diff[indices]))
    return float(np.sum(diff[indices]))


def _block_bootstrap_indices(
    rng: np.random.Generator,
    n_steps: int,
    block_size: int,
) -> np.ndarray:
    if n_steps <= 0:
        return np.asarray([], dtype=np.int64)
    block_size = max(1, min(int(block_size), n_steps))
    starts = rng.integers(0, n_steps, size=int(np.ceil(n_steps / block_size)))
    blocks = [(start + np.arange(block_size)) % n_steps for start in starts]
    return np.concatenate(blocks).astype(np.int64)[:n_steps]


def _paired_block_bootstrap_significance(
    candidate_step_results: pd.DataFrame,
    baseline_step_results: pd.DataFrame,
    bootstrap_samples: int,
    block_size: int,
    seed: int,
    confidence_level: float,
    alpha: float,
) -> dict[str, float | int]:
    metric_deltas = _paired_metric_deltas(candidate_step_results, baseline_step_results)
    n_steps = len(next(iter(metric_deltas.values()))) if metric_deltas else 0
    metric_names = (
        "total_reward",
        "avg_reward_per_env_step",
        "action_frequency",
        "event_trigger_rate",
        "E_daily_kwh_per_day",
    )
    result: dict[str, float | int] = {
        "bootstrap_paired_steps": int(n_steps),
        "bootstrap_samples": int(max(0, bootstrap_samples)),
        "bootstrap_block_size": int(max(1, block_size)),
        "bootstrap_confidence_level": float(confidence_level),
        "bootstrap_significance_level": float(alpha),
    }
    if n_steps <= 0 or bootstrap_samples <= 0:
        for metric_name in metric_names:
            result[f"bootstrap_{metric_name}_delta_mean"] = 0.0
            result[f"bootstrap_{metric_name}_delta_ci_lower"] = 0.0
            result[f"bootstrap_{metric_name}_delta_ci_upper"] = 0.0
            result[f"bootstrap_{metric_name}_delta_p_value"] = 1.0
            result[f"bootstrap_{metric_name}_delta_significant"] = 0
        return result

    rng = np.random.default_rng(seed)
    lower_q = (1.0 - confidence_level) / 2.0
    upper_q = 1.0 - lower_q
    for metric_name in metric_names:
        diff = metric_deltas[metric_name]
        observed_delta = _bootstrap_statistic(diff, metric_name, np.arange(n_steps, dtype=np.int64))
        bootstrap_values = np.asarray(
            [
                _bootstrap_statistic(
                    diff,
                    metric_name,
                    _block_bootstrap_indices(rng, n_steps=n_steps, block_size=block_size),
                )
                for _ in range(int(bootstrap_samples))
            ],
            dtype=np.float64,
        )
        centered = bootstrap_values - observed_delta
        p_value = float(np.mean(np.abs(centered) >= abs(observed_delta)))
        ci_lower = float(np.quantile(bootstrap_values, lower_q))
        ci_upper = float(np.quantile(bootstrap_values, upper_q))
        result[f"bootstrap_{metric_name}_delta_mean"] = float(observed_delta)
        result[f"bootstrap_{metric_name}_delta_ci_lower"] = ci_lower
        result[f"bootstrap_{metric_name}_delta_ci_upper"] = ci_upper
        result[f"bootstrap_{metric_name}_delta_p_value"] = float(min(1.0, max(0.0, p_value)))
        result[f"bootstrap_{metric_name}_delta_significant"] = int((ci_lower > 0.0 or ci_upper < 0.0) and p_value < alpha)

    return result


def _numeric_column_values(step_results: pd.DataFrame, column: str, length: int, default: float = 0.0) -> np.ndarray:
    if column not in step_results.columns:
        return np.full(length, default, dtype=np.float64)
    return step_results[column].to_numpy(dtype=np.float64)[:length]


def _gate_trace_diagnostics(candidate_step_results: pd.DataFrame, baseline_step_results: pd.DataFrame) -> dict[str, float | int]:
    common_length = min(len(candidate_step_results), len(baseline_step_results))
    diagnostics: dict[str, float | int] = {"diagnostic_paired_steps": int(common_length)}
    if common_length <= 0:
        return {
            **diagnostics,
            "gate_signal_diff_count": 0,
            "gate_signal_same_rate": 1.0,
            "action_update_diff_count": 0,
            "action_update_same_rate": 1.0,
            "action_value_diff_count": 0,
            "action_value_same_rate": 1.0,
        }

    for column, prefix in (
        ("gate_signal", "gate_signal"),
        ("action_updated", "action_update"),
        ("action_value", "action_value"),
    ):
        candidate_values = _numeric_column_values(candidate_step_results, column, common_length)
        baseline_values = _numeric_column_values(baseline_step_results, column, common_length)
        diff_count = int(np.sum(~np.isclose(candidate_values, baseline_values, atol=ZERO_SENSITIVITY_ABS_TOL)))
        diagnostics[f"{prefix}_diff_count"] = diff_count
        diagnostics[f"{prefix}_same_rate"] = float(1.0 - diff_count / common_length)

    for column in (
        "anomaly_score",
        "adaptive_threshold",
        "gate_base_threshold",
        "gate_trigger_threshold",
        "gate_reset_threshold",
        "gate_decision_margin",
        "gate_trigger_state",
        "gate_min_interval_satisfied",
    ):
        candidate_values = _numeric_column_values(candidate_step_results, column, common_length, default=np.nan)
        baseline_values = _numeric_column_values(baseline_step_results, column, common_length, default=np.nan)
        valid_mask = np.isfinite(candidate_values) & np.isfinite(baseline_values)
        prefix = column.replace("gate_", "")
        if not np.any(valid_mask):
            diagnostics[f"candidate_{prefix}_mean"] = float("nan")
            diagnostics[f"candidate_{prefix}_std"] = float("nan")
            diagnostics[f"{prefix}_delta_mean"] = float("nan")
            diagnostics[f"{prefix}_delta_abs_mean"] = float("nan")
            diagnostics[f"{prefix}_delta_abs_max"] = float("nan")
            continue

        candidate_valid = candidate_values[valid_mask]
        delta = candidate_valid - baseline_values[valid_mask]
        diagnostics[f"candidate_{prefix}_mean"] = float(np.mean(candidate_valid))
        diagnostics[f"candidate_{prefix}_std"] = float(np.std(candidate_valid))
        diagnostics[f"{prefix}_delta_mean"] = float(np.mean(delta))
        diagnostics[f"{prefix}_delta_abs_mean"] = float(np.mean(np.abs(delta)))
        diagnostics[f"{prefix}_delta_abs_max"] = float(np.max(np.abs(delta)))

    return diagnostics


def _format_zero_sensitivity_note(
    spec: SensitivitySpec,
    sorted_rows: list[dict[str, Any]],
    prewarm_sample_count: int,
) -> tuple[int, str]:
    reward_span = float(max(row["total_reward"] for row in sorted_rows) - min(row["total_reward"] for row in sorted_rows))
    action_frequency_span = float(max(row["action_frequency"] for row in sorted_rows) - min(row["action_frequency"] for row in sorted_rows))
    event_trigger_rate_span = float(max(row["event_trigger_rate"] for row in sorted_rows) - min(row["event_trigger_rate"] for row in sorted_rows))
    energy_span = float(max(row["E_daily_kwh_per_day"] for row in sorted_rows) - min(row["E_daily_kwh_per_day"] for row in sorted_rows))
    is_zero = int(
        reward_span <= ZERO_SENSITIVITY_ABS_TOL
        and action_frequency_span <= ZERO_SENSITIVITY_ABS_TOL
        and event_trigger_rate_span <= ZERO_SENSITIVITY_ABS_TOL
        and energy_span <= ZERO_SENSITIVITY_ABS_TOL
    )
    if not is_zero:
        return 0, "非零敏感：至少一个候选值改变了奖励、能耗或触发/动作频率。"

    max_gate_signal_diff_count = max(int(row.get("gate_signal_diff_count", 0)) for row in sorted_rows)
    max_action_update_diff_count = max(int(row.get("action_update_diff_count", 0)) for row in sorted_rows)
    max_action_value_diff_count = max(int(row.get("action_value_diff_count", 0)) for row in sorted_rows)
    max_score_delta_abs = max(float(row.get("anomaly_score_delta_abs_mean", 0.0)) for row in sorted_rows)
    max_threshold_delta_abs = max(float(row.get("adaptive_threshold_delta_abs_mean", 0.0)) for row in sorted_rows)

    if spec.name == "threshold_min_samples_for_optimization":
        max_candidate = max(int(round(float(value))) for value in spec.values)
        if prewarm_sample_count >= max_candidate:
            return (
                1,
                f"零敏感验证：预热样本数 {prewarm_sample_count} >= 最大候选值 {max_candidate}，测试开始前所有候选都已满足阈值优化启动条件；触发/动作轨迹未产生有效差异。",
            )

    if max_gate_signal_diff_count == 0 and max_action_update_diff_count == 0 and max_action_value_diff_count == 0:
        return 1, "零敏感验证：所有候选的 gate_signal、action_updated 与 action_value 轨迹均与默认门控一致，因此奖励与能耗完全一致。"
    if max_action_update_diff_count == 0 and max_action_value_diff_count == 0:
        return 1, "零敏感验证：门控内部分数/阈值可能变化，但未改变动作更新和值轨迹，因此控制指标不变。"
    return (
        1,
        "零敏感验证：最终指标跨度低于数值容差；"
        f"最大平均分数差={max_score_delta_abs:.3e}，最大平均阈值差={max_threshold_delta_abs:.3e}。",
    )


def _build_candidate_gate(
    config: type[Any],
    test_data: pd.DataFrame,
    prewarm_features: np.ndarray,
) -> Any:
    gate = create_streaming_gate(
        config=config,
        test_data=test_data,
        logger=_SilentLogger(),
        gate_state_path=None,
    )
    _prewarm_gate(gate, prewarm_features)
    return gate


def _evaluate_once(
    config: type[Any],
    agent: Any,
    action_space: np.ndarray,
    reward_calc: Any,
    test_data: pd.DataFrame,
    prewarm_features: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    gate = _build_candidate_gate(config, test_data, prewarm_features)
    env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
    summary, step_results = test_event_driven(
        agent=agent,
        env=env,
        data=test_data,
        action_space=action_space,
        gate=gate,
        feature_columns=config.FEATURE_COLUMNS,
        supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
    )
    return summary, step_results


def run_gate_parameter_sensitivity_analysis(
    train_experiment_dir: Path | None = None,
    output_dir: Path | None = None,
    data_split: str = "test",
    parameters: list[str] | None = None,
    bootstrap_samples: int | None = None,
    bootstrap_block_size: int | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    base_config = ControlCompareConfig
    resolved_train_dir = resolve_train_experiment_dir(train_experiment_dir)
    resolved_parameters = _resolve_parameters(parameters)
    resolved_bootstrap_samples = int(
        bootstrap_samples
        if bootstrap_samples is not None
        else getattr(base_config, "GATE_SENSITIVITY_BOOTSTRAP_SAMPLES", 500)
    )
    resolved_bootstrap_block_size = int(
        bootstrap_block_size
        if bootstrap_block_size is not None
        else getattr(base_config, "GATE_SENSITIVITY_BOOTSTRAP_BLOCK_SIZE", 48)
    )
    resolved_bootstrap_seed = int(
        bootstrap_seed
        if bootstrap_seed is not None
        else getattr(base_config, "GATE_SENSITIVITY_BOOTSTRAP_SEED", 42)
    )
    confidence_level = float(getattr(base_config, "GATE_SENSITIVITY_CONFIDENCE_LEVEL", 0.95))
    confidence_level = float(np.clip(confidence_level, 1e-6, 1.0 - 1e-6))
    significance_level = float(getattr(base_config, "GATE_SENSITIVITY_SIGNIFICANCE_LEVEL", 0.05))
    significance_level = float(np.clip(significance_level, 1e-6, 1.0 - 1e-6))

    default_output_root = base_config.LOG_ROOT_DIR / "online_anomaly_detection" / "sensitivity"
    analysis_root = Path(output_dir or default_output_root) / base_config.TIMESTAMP
    analysis_root.mkdir(parents=True, exist_ok=True)

    context = create_experiment_context(
        experiment_dir=analysis_root,
        config=base_config,
        save_config=False,
        log_filename=base_config.EVALUATION_LOG_FILENAME,
        metrics_filename=base_config.EVALUATION_METRICS_FILENAME,
        config_filename=base_config.CONFIG_FILENAME,
    )
    logger = context.logger
    metrics_recorder = context.metrics_recorder

    logger.info("=" * 70)
    logger.info("超参敏感性分析")
    logger.info("=" * 70)
    logger.info(f"训练实验目录: {resolved_train_dir}")
    logger.info(f"数据划分: {data_split}")
    logger.info(f"分析参数: {[spec.name for spec in resolved_parameters]}")
    logger.info(
        "Bootstrap 显著性: "
        f"samples={resolved_bootstrap_samples}, "
        f"block_size={resolved_bootstrap_block_size}, "
        f"seed={resolved_bootstrap_seed}, "
        f"confidence={confidence_level:.3f}, "
        f"alpha={significance_level:.3f}"
    )
    logger.info(f"输出目录: {analysis_root}")

    test_data, action_space, agent, checkpoint, reward_calc = build_test_components(
        config=base_config,
        resolved_train_dir=resolved_train_dir,
        data_split=data_split,
    )
    prewarm_features = _load_prewarm_features(base_config)
    prewarm_sample_count = int(len(prewarm_features))

    baseline_config = _build_analysis_config(base_config, {})
    baseline_summary, baseline_step_results = _evaluate_once(
        config=baseline_config,
        agent=agent,
        action_space=action_space,
        reward_calc=reward_calc,
        test_data=test_data,
        prewarm_features=prewarm_features,
    )
    baseline_env = SequenceEnv(test_data, base_config.STATE_COLUMNS, reward_calc)
    fixed_baseline_summary, _ = test_fixed_interval(
        agent=agent,
        env=baseline_env,
        action_space=action_space,
        fixed_interval=int(base_config.GATE_OPT_BASELINE_FIXED_INTERVAL),
        supply_temp_ref=base_config.CHILLER_SUPPLY_TEMP_REF,
        comfort_lower_bound=base_config.COMFORT_LOWER_BOUND,
        comfort_upper_bound=base_config.COMFORT_UPPER_BOUND,
    )
    baseline_metrics = _metric_summary(
        baseline_summary,
        baseline_step_results,
        baseline_summary,
        fixed_baseline_summary,
    )

    logger.info(
        "基线结果: "
        f"reward={baseline_metrics['total_reward']:.4f}, "
        f"action_rate={baseline_metrics['action_frequency']:.4f}, "
        f"event_trigger_rate={baseline_metrics['event_trigger_rate']:.4f}, "
        f"E_daily={baseline_metrics['E_daily_kwh_per_day']:.4f}"
    )

    result_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in resolved_parameters:
        logger.info(f"扫描参数 {spec.name}: {list(spec.values)}")
        per_param_rows: list[dict[str, Any]] = []

        for trial_index, candidate_value in enumerate(spec.values, start=1):
            overrides = _build_candidate_overrides(base_config, spec, candidate_value)
            candidate_config = _build_analysis_config(base_config, overrides)
            summary, step_results = _evaluate_once(
                config=candidate_config,
                agent=agent,
                action_space=action_space,
                reward_calc=reward_calc,
                test_data=test_data,
                prewarm_features=prewarm_features,
            )
            metrics = _metric_summary(
                summary,
                step_results,
                baseline_summary,
                fixed_baseline_summary,
            )
            significance = _paired_block_bootstrap_significance(
                candidate_step_results=step_results,
                baseline_step_results=baseline_step_results,
                bootstrap_samples=resolved_bootstrap_samples,
                block_size=resolved_bootstrap_block_size,
                seed=resolved_bootstrap_seed,
                confidence_level=confidence_level,
                alpha=significance_level,
            )
            diagnostics = _gate_trace_diagnostics(
                candidate_step_results=step_results,
                baseline_step_results=baseline_step_results,
            )
            row = {
                "parameter": spec.name,
                "parameter_label": spec.label,
                "group": spec.group,
                "trial_index": trial_index,
                "candidate_value": candidate_value,
                "candidate_value_numeric": float(candidate_value),
                "is_baseline_value": bool(np.isclose(float(candidate_value), float(getattr(base_config, spec.config_attr)), atol=1e-12)),
                **metrics,
                **significance,
                **diagnostics,
            }
            result_rows.append(row)
            per_param_rows.append(row)

        sorted_rows = sorted(per_param_rows, key=lambda item: float(item["candidate_value_numeric"]))
        is_zero_sensitivity, zero_sensitivity_note = _format_zero_sensitivity_note(
            spec=spec,
            sorted_rows=sorted_rows,
            prewarm_sample_count=prewarm_sample_count,
        )
        summary_rows.append(
            {
                "parameter": spec.name,
                "parameter_label": spec.label,
                "group": spec.group,
                "baseline_value": float(getattr(base_config, spec.config_attr)),
                "candidate_values": json.dumps([row["candidate_value"] for row in sorted_rows], ensure_ascii=False),
                "best_total_reward_value": max(sorted_rows, key=lambda item: float(item["total_reward"]))["candidate_value"],
                "closest_to_default_gate_action_frequency_value": min(
                    sorted_rows,
                    key=lambda item: abs(float(item["action_frequency"]) - float(baseline_metrics["action_frequency"])),
                )["candidate_value"],
                "reward_span": float(max(row["total_reward"] for row in sorted_rows) - min(row["total_reward"] for row in sorted_rows)),
                "action_frequency_span": float(max(row["action_frequency"] for row in sorted_rows) - min(row["action_frequency"] for row in sorted_rows)),
                "event_trigger_rate_span": float(max(row["event_trigger_rate"] for row in sorted_rows) - min(row["event_trigger_rate"] for row in sorted_rows)),
                "E_daily_kwh_per_day_span": float(
                    max(row["E_daily_kwh_per_day"] for row in sorted_rows) - min(row["E_daily_kwh_per_day"] for row in sorted_rows)
                ),
                "significant_total_reward_candidate_count": int(
                    sum(int(row.get("bootstrap_total_reward_delta_significant", 0)) for row in sorted_rows)
                ),
                "significant_action_frequency_candidate_count": int(
                    sum(int(row.get("bootstrap_action_frequency_delta_significant", 0)) for row in sorted_rows)
                ),
                "significant_event_trigger_rate_candidate_count": int(
                    sum(int(row.get("bootstrap_event_trigger_rate_delta_significant", 0)) for row in sorted_rows)
                ),
                "significant_E_daily_candidate_count": int(
                    sum(int(row.get("bootstrap_E_daily_kwh_per_day_delta_significant", 0)) for row in sorted_rows)
                ),
                "min_total_reward_delta_p_value": float(
                    min(float(row.get("bootstrap_total_reward_delta_p_value", 1.0)) for row in sorted_rows)
                ),
                "max_gate_signal_diff_count": int(max(int(row.get("gate_signal_diff_count", 0)) for row in sorted_rows)),
                "max_action_update_diff_count": int(max(int(row.get("action_update_diff_count", 0)) for row in sorted_rows)),
                "max_action_value_diff_count": int(max(int(row.get("action_value_diff_count", 0)) for row in sorted_rows)),
                "is_zero_sensitivity": int(is_zero_sensitivity),
                "zero_sensitivity_note": zero_sensitivity_note,
            }
        )

    results_df = pd.DataFrame(result_rows)
    summary_df = pd.DataFrame(summary_rows)

    results_path = analysis_root / "gate_parameter_sensitivity_results.csv"
    summary_path = analysis_root / "gate_parameter_sensitivity_summary.csv"
    diagnostics_path = analysis_root / "gate_parameter_sensitivity_diagnostics.csv"
    baseline_path = analysis_root / "gate_parameter_sensitivity_baseline.json"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    diagnostic_columns = [
        column
        for column in results_df.columns
        if column
        in {
            "parameter",
            "parameter_label",
            "group",
            "trial_index",
            "candidate_value",
            "is_baseline_value",
        }
        or column.startswith("bootstrap_")
        or column.endswith("_diff_count")
        or column.endswith("_same_rate")
        or column.endswith("_delta_mean")
        or column.endswith("_delta_abs_mean")
        or column.endswith("_delta_abs_max")
        or column.startswith("candidate_")
    ]
    results_df[diagnostic_columns].to_csv(diagnostics_path, index=False)

    payload = {
        "train_experiment_dir": str(resolved_train_dir),
        "data_split": data_split,
        "parameters": [spec.name for spec in resolved_parameters],
        "bootstrap": {
            "samples": resolved_bootstrap_samples,
            "block_size": resolved_bootstrap_block_size,
            "seed": resolved_bootstrap_seed,
            "confidence_level": confidence_level,
            "significance_level": significance_level,
        },
        "baseline": baseline_metrics,
        "best_epoch_from_train": int(checkpoint.get("epoch", -1)) + 1,
        "results_csv": str(results_path),
        "summary_csv": str(summary_path),
        "diagnostics_csv": str(diagnostics_path),
    }
    with open(baseline_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    metrics_recorder.save_metrics(payload)

    logger.info(f"结果已保存: {results_path}")
    logger.info(f"摘要已保存: {summary_path}")
    logger.info(f"诊断已保存: {diagnostics_path}")
    logger.info(f"基线已保存: {baseline_path}")

    return payload
