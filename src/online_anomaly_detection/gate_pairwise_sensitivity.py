from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.control_compare_config import ControlCompareConfig
from src.control_evaluation.common import build_test_components, create_streaming_gate, resolve_train_experiment_dir
from src.control_evaluation.strategies.event_driven import test_event_driven
from src.control_evaluation.strategies.fixed_interval import test_fixed_interval
from ml_toolkit.rl import SequenceEnv
from ml_toolkit.utils import create_experiment_context

from src.online_anomaly_detection.gate_sensitivity_analysis import (
    DEFAULT_SPECS,
    SPEC_BY_NAME,
    _SilentLogger,
    _build_analysis_config,
    _build_candidate_overrides,
    _evaluate_once,
    _load_prewarm_features,
    _metric_summary,
)


PAIRWISE_SPECS: dict[str, dict[str, Any]] = {
    # (label, label, metric_keys)
    "ratio": {
        "x": "threshold_bias",
        "y": "trigger_hysteresis_margin",
        "label_x": "$b_{\\mathrm{bias}}$",
        "label_y": "$m_{\\mathrm{hys}}$",
    },
    "timescale": {
        "x": "local_window_size",
        "y": "score_short_weight",
        "label_x": "$W$",
        "label_y": "$w_{\\mathrm{s}}$",
    },
}

# 默认候选值：
# ratio group: 5 x 5 = 25 combos (low-cost)
# timescale group: 6 x 5 = 30 combos (low-cost)
DEFAULT_RATIO_X: tuple[float, ...] = (-0.12, -0.08, -0.043, 0.0, 0.04)
DEFAULT_RATIO_Y: tuple[float, ...] = (0.0, 0.01, 0.022, 0.05, 0.08)
DEFAULT_TIMING_X: tuple[int, ...] = (30, 60, 90, 135, 180, 240)
DEFAULT_TIMING_Y: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75)


def run_gate_pairwise_sensitivity(
    train_experiment_dir: Path | None = None,
    output_dir: Path | None = None,
    data_split: str = "test",
    group: str = "ratio",
    bootstrap_samples: int | None = None,
    bootstrap_block_size: int | None = None,
    bootstrap_seed: int | None = None,
) -> Path:
    if group not in PAIRWISE_SPECS:
        raise ValueError(f"未知分组: {group}，可选: {list(PAIRWISE_SPECS.keys())}")

    spec = PAIRWISE_SPECS[group]
    x_name, y_name = spec["x"], spec["y"]
    x_spec, y_spec = SPEC_BY_NAME[x_name], SPEC_BY_NAME[y_name]

    if group == "ratio":
        x_values = DEFAULT_RATIO_X
        y_values = DEFAULT_RATIO_Y
    else:
        x_values = DEFAULT_TIMING_X
        y_values = DEFAULT_TIMING_Y

    base_config = ControlCompareConfig
    resolved_train_dir = resolve_train_experiment_dir(train_experiment_dir)

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
    significance_level = float(getattr(base_config, "GATE_SENSITIVITY_SIGNIFICANCE_LEVEL", 0.05))

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
    logger.info(f"二维敏感性分组: {group}")
    logger.info(f"X: {x_name} ({list(x_values)})")
    logger.info(f"Y: {y_name} ({list(y_values)})")
    logger.info(f"总组合数: {len(x_values) * len(y_values)}")
    logger.info("=" * 70)

    test_data, action_space, agent, checkpoint, reward_calc = build_test_components(
        config=base_config,
        resolved_train_dir=resolved_train_dir,
        data_split=data_split,
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

    prewarm_features = _load_prewarm_features(base_config)

    result_rows: list[dict[str, Any]] = []
    total_combos = len(x_values) * len(y_values)
    combo_index = 0

    for x_val in x_values:
        for y_val in y_values:
            combo_index += 1
            x_overrides = _build_candidate_overrides(base_config, x_spec, x_val)
            y_overrides = _build_candidate_overrides(base_config, y_spec, y_val)
            merged_overrides = {**x_overrides, **y_overrides}
            candidate_config = _build_analysis_config(base_config, merged_overrides)

            summary, step_results = _evaluate_once(
                config=candidate_config,
                agent=agent,
                action_space=action_space,
                reward_calc=reward_calc,
                test_data=test_data,
                prewarm_features=prewarm_features,
            )
            metrics = _metric_summary(summary, step_results, summary, fixed_baseline_summary)

            is_baseline_x = bool(np.isclose(float(x_val), float(getattr(base_config, x_spec.config_attr)), atol=1e-12))
            is_baseline_y = bool(np.isclose(float(y_val), float(getattr(base_config, y_spec.config_attr)), atol=1e-12))

            row = {
                "group": group,
                "x_param": x_name,
                "y_param": y_name,
                "x_value": float(x_val),
                "y_value": float(y_val),
                "x_is_baseline": int(is_baseline_x),
                "y_is_baseline": int(is_baseline_y),
                **metrics,
            }
            result_rows.append(row)

            logger.info(
                f"[{combo_index}/{total_combos}] "
                f"{x_name}={x_val}, {y_name}={y_val} | "
                f"reward={metrics['total_reward']:.4f}, "
                f"N_daily={metrics['N_daily_count_per_day']:.2f}, "
                f"E_daily={metrics['E_daily_kwh_per_day']:.2f}"
            )

    results_df = pd.DataFrame(result_rows)
    results_path = analysis_root / f"gate_pairwise_{group}_results.csv"
    results_df.to_csv(results_path, index=False)

    payload = {
        "train_experiment_dir": str(resolved_train_dir),
        "data_split": data_split,
        "group": group,
        "x_param": x_name,
        "y_param": y_name,
        "x_values": [float(v) for v in x_values],
        "y_values": [float(v) for v in y_values],
        "total_combos": total_combos,
        "results_csv": str(results_path),
    }
    meta_path = analysis_root / f"gate_pairwise_{group}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    metrics_recorder.save_metrics(payload)
    logger.info(f"结果已保存: {results_path}")
    return results_path
