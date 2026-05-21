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
    SensitivitySpec(
        name="threshold_bias",
        config_attr="GATE_THRESHOLD_BIAS",
        label="Threshold Bias",
        values=(-0.20, -0.10, -0.043057734706757875, -0.02, 0.0, 0.05),
        group="threshold",
    ),
    SensitivitySpec(
        name="threshold_quantile",
        config_attr="THRESHOLD_QUANTILE",
        label="Threshold Quantile",
        values=(0.55, 0.60, 0.6675415126464518, 0.75, 0.85, 0.90),
        group="threshold",
    ),
    SensitivitySpec(
        name="threshold_mad_scale",
        config_attr="THRESHOLD_MAD_SCALE",
        label="MAD Scale",
        values=(0.8, 1.0, 1.1352750910681282, 1.5, 2.0, 2.8),
        group="threshold",
    ),
    SensitivitySpec(
        name="threshold_local_update_rate",
        config_attr="THRESHOLD_LOCAL_UPDATE_RATE",
        label="Local Update Rate",
        values=(0.10, 0.25, 0.40, 0.5956588562510114, 0.70, 0.80),
        group="threshold",
    ),
    SensitivitySpec(
        name="contamination",
        config_attr="GATE_CONTAMINATION",
        label="Contamination",
        values=(0.05, 0.10, 0.20, 0.3104487614924219, 0.31, 0.35),
        group="threshold",
    ),
    SensitivitySpec(
        name="trigger_hysteresis_margin",
        config_attr="GATE_TRIGGER_HYSTERESIS_MARGIN",
        label="Hysteresis Margin",
        values=(0.0, 0.01, 0.021759826420260316, 0.04, 0.08, 0.12),
        group="threshold",
    ),
    SensitivitySpec(
        name="local_window_size",
        config_attr="GATE_LOCAL_WINDOW_SIZE",
        label="Local Window Size",
        values=(30, 60, 135, 180, 227, 240),
        group="statistical",
        integer=True,
    ),
    SensitivitySpec(
        name="global_ema_decay",
        config_attr="GATE_GLOBAL_EMA_DECAY",
        label="Global EMA Decay",
        values=(0.005, 0.02, 0.0625, 0.0745, 0.0819, 0.12),
        group="statistical",
    ),
    SensitivitySpec(
        name="reference_samples",
        config_attr="GATE_REFERENCE_SAMPLES",
        label="Reference Samples",
        values=(200, 500, 700, 910, 1100, 1200),
        group="statistical",
        integer=True,
    ),
    SensitivitySpec(
        name="threshold_min_samples_for_optimization",
        config_attr="THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION",
        label="Threshold Min Samples",
        values=(20, 50, 70, 104, 120),
        group="statistical",
        integer=True,
    ),
    SensitivitySpec(
        name="score_short_weight",
        config_attr="GATE_SCORE_SHORT_WEIGHT",
        label="Short Weight",
        values=(0.30, 0.40, 0.50, 0.55, 0.60, 0.65),
        group="fusion",
    ),
    SensitivitySpec(
        name="score_medium_weight",
        config_attr="GATE_SCORE_MEDIUM_WEIGHT",
        label="Medium Weight",
        values=(0.05, 0.10, 0.15, 0.20, 0.275, 0.30),
        group="fusion",
    ),
    SensitivitySpec(
        name="threshold_quantile_weight",
        config_attr="THRESHOLD_QUANTILE_WEIGHT",
        label="Quantile Weight",
        values=(0.20, 0.35, 0.575, 0.70, 0.85, 0.95),
        group="threshold",
    ),
)

DEFAULT_PARAM_ORDER: tuple[str, ...] = (
    "threshold_bias",
    "threshold_quantile",
    "threshold_mad_scale",
    "threshold_local_update_rate",
    "contamination",
    "trigger_hysteresis_margin",
    "local_window_size",
    "global_ema_decay",
    "reference_samples",
    "threshold_min_samples_for_optimization",
)
SPEC_BY_NAME: dict[str, SensitivitySpec] = {spec.name: spec for spec in DEFAULT_SPECS}
PREWARM_SPLITS: tuple[str, ...] = ("train", "val")


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

    reward_drop_ratio = (baseline_reward - total_reward) / max(abs(baseline_reward), 1e-8)
    energy_increase_ratio = (energy_daily - baseline_energy_daily) / max(abs(baseline_energy_daily), 1e-8)

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
        "delta_total_reward_vs_default": float(total_reward - default_reward),
        "delta_action_frequency_vs_default": float(float(summary.get("action_frequency", 0.0)) - default_action_frequency),
        "delta_event_trigger_rate_vs_default": float(float(summary.get("event_trigger_rate", event_trigger_rate)) - default_event_trigger_rate),
        "delta_E_daily_kwh_per_day_vs_default": float(energy_daily - default_energy_daily),
        "reward_drop_ratio_vs_baseline": float(reward_drop_ratio),
        "energy_increase_ratio_vs_baseline": float(energy_increase_ratio),
    }


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
) -> dict[str, Any]:
    base_config = ControlCompareConfig
    resolved_train_dir = resolve_train_experiment_dir(train_experiment_dir)
    resolved_parameters = _resolve_parameters(parameters)

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
    logger.info(f"输出目录: {analysis_root}")

    test_data, action_space, agent, checkpoint, reward_calc = build_test_components(
        config=base_config,
        resolved_train_dir=resolved_train_dir,
        data_split=data_split,
    )
    prewarm_features = _load_prewarm_features(base_config)

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
            row = {
                "parameter": spec.name,
                "parameter_label": spec.label,
                "group": spec.group,
                "trial_index": trial_index,
                "candidate_value": candidate_value,
                "candidate_value_numeric": float(candidate_value),
                "is_baseline_value": bool(np.isclose(float(candidate_value), float(getattr(base_config, spec.config_attr)), atol=1e-12)),
                **metrics,
            }
            result_rows.append(row)
            per_param_rows.append(row)

        sorted_rows = sorted(per_param_rows, key=lambda item: float(item["candidate_value_numeric"]))
        summary_rows.append(
            {
                "parameter": spec.name,
                "parameter_label": spec.label,
                "group": spec.group,
                "baseline_value": float(getattr(base_config, spec.config_attr)),
                "candidate_values": json.dumps([row["candidate_value"] for row in sorted_rows], ensure_ascii=False),
                "best_total_reward_value": max(sorted_rows, key=lambda item: float(item["total_reward"]))["candidate_value"],
                "best_action_frequency_value": min(
                    sorted_rows,
                    key=lambda item: abs(float(item["action_frequency"]) - float(baseline_metrics["action_frequency"])),
                )["candidate_value"],
                "reward_span": float(max(row["total_reward"] for row in sorted_rows) - min(row["total_reward"] for row in sorted_rows)),
                "action_frequency_span": float(max(row["action_frequency"] for row in sorted_rows) - min(row["action_frequency"] for row in sorted_rows)),
                "event_trigger_rate_span": float(max(row["event_trigger_rate"] for row in sorted_rows) - min(row["event_trigger_rate"] for row in sorted_rows)),
            }
        )

    results_df = pd.DataFrame(result_rows)
    summary_df = pd.DataFrame(summary_rows)

    results_path = analysis_root / "gate_parameter_sensitivity_results.csv"
    summary_path = analysis_root / "gate_parameter_sensitivity_summary.csv"
    baseline_path = analysis_root / "gate_parameter_sensitivity_baseline.json"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "train_experiment_dir": str(resolved_train_dir),
        "data_split": data_split,
        "parameters": [spec.name for spec in resolved_parameters],
        "baseline": baseline_metrics,
        "best_epoch_from_train": int(checkpoint.get("epoch", -1)) + 1,
        "results_csv": str(results_path),
        "summary_csv": str(summary_path),
    }
    with open(baseline_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    metrics_recorder.save_metrics(payload)

    logger.info(f"结果已保存: {results_path}")
    logger.info(f"摘要已保存: {summary_path}")
    logger.info(f"基线已保存: {baseline_path}")

    return payload
