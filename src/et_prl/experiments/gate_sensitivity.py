from __future__ import annotations

import copy
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from et_prl.environments import SequenceEnv
from et_prl.utils import create_experiment_context

from et_prl.config.control_compare import ControlCompareConfig
from et_prl.evaluation.control.common import (
    _create_reward_calculator,
    build_test_components,
    create_streaming_gate,
    resolve_train_experiment_dir,
)
from et_prl.evaluation.control.strategies.event_driven import test_event_driven
from et_prl.evaluation.control.strategies.fixed_interval import test_fixed_interval
from et_prl.config.loader import load_config


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SensitivitySpec:
    name: str
    config_attr: str
    label: str
    group: str
    integer: bool = False


@dataclass(frozen=True)
class PairwiseDesign:
    name: str
    label: str
    x_param: str
    y_param: str
    x_values: tuple[Any, ...]
    y_values: tuple[Any, ...]


@dataclass(frozen=True)
class SingleSweepDesign:
    name: str
    label: str
    parameter: str
    values: tuple[Any, ...]


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter specifications
# ═══════════════════════════════════════════════════════════════════════════════

SPECS: tuple[SensitivitySpec, ...] = (
    SensitivitySpec("threshold_bias", "GATE_THRESHOLD_BIAS", "b_bias", "boundary"),
    SensitivitySpec("trigger_hysteresis_margin", "GATE_TRIGGER_HYSTERESIS_MARGIN", "m_hys", "boundary"),
    SensitivitySpec("local_window_size", "GATE_LOCAL_WINDOW_SIZE", "W", "timescale", integer=True),
    SensitivitySpec("score_short_weight", "GATE_SCORE_SHORT_WEIGHT", "w_s", "timescale"),
    SensitivitySpec("threshold_quantile", "THRESHOLD_QUANTILE", "q", "threshold"),
)
SPEC_BY_NAME: dict[str, SensitivitySpec] = {s.name: s for s in SPECS}

# (a) Mechanism-coupled: b_bias × m_hys
# (b) Mechanism-coupled: W × w_s
# (c) Single-factor sweep: q
PAIRWISE_DESIGNS: tuple[PairwiseDesign, ...] = (
    PairwiseDesign(
        name="boundary",
        label="b_bias x m_hys",
        x_param="threshold_bias",
        y_param="trigger_hysteresis_margin",
        x_values=(-0.12, -0.08, -0.043, 0.0, 0.04),
        y_values=(0.0, 0.01, 0.022, 0.05, 0.08),
    ),
    PairwiseDesign(
        name="timescale",
        label="W x w_s",
        x_param="local_window_size",
        y_param="score_short_weight",
        x_values=(30, 60, 90, 135, 180, 240),
        y_values=(0.35, 0.45, 0.55, 0.65, 0.75),
    ),
)

SINGLE_SWEEPS: tuple[SingleSweepDesign, ...] = (
    SingleSweepDesign(
        name="quantile",
        label="q sweep",
        parameter="threshold_quantile",
        values=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    ),
)

PREWARM_SPLITS: tuple[str, ...] = ("train", "val")
ZERO_SENSITIVITY_ABS_TOL = 1e-9
SAMPLE_INTERVAL_MINUTES = 5.0

ROUND_METRIC_COLUMNS: tuple[str, ...] = (
    "total_reward",
    "avg_reward_per_action",
    "avg_reward_per_env_step",
    "avg_energy_score",
    "avg_comfort_score",
    "action_count",
    "action_frequency",
    "N_daily_count_per_day",
    "event_trigger_count",
    "event_trigger_rate",
    "E_daily_kwh_per_day",
    "delta_total_reward_vs_default_gate",
    "delta_action_frequency_vs_default_gate",
    "delta_N_daily_count_per_day_vs_default_gate",
    "delta_event_trigger_rate_vs_default_gate",
    "delta_E_daily_kwh_per_day_vs_default_gate",
    "signed_reward_gap_ratio_vs_fixed_baseline",
    "signed_energy_change_ratio_vs_fixed_baseline",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class _SilentLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        return None
    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None
    def error(self, *args: Any, **kwargs: Any) -> None:
        return None


def _coerce_value(spec: SensitivitySpec, value: Any) -> Any:
    if spec.integer:
        return int(round(float(value)))
    return float(value)


def _is_baseline_value(base_config: type[Any], spec: SensitivitySpec, value: Any) -> bool:
    base = _coerce_value(spec, getattr(base_config, spec.config_attr))
    if spec.integer:
        return int(value) == int(base)
    return bool(np.isclose(float(value), float(base), rtol=0.0, atol=1e-12))


def _values_with_baseline(
    base_config: type[Any], spec_name: str, values: tuple[Any, ...]
) -> tuple[Any, ...]:
    spec = SPEC_BY_NAME[spec_name]
    prepared = [_coerce_value(spec, v) for v in values]
    prepared.append(_coerce_value(spec, getattr(base_config, spec.config_attr)))
    unique: list[Any] = []
    for v in sorted(prepared, key=lambda x: float(x)):
        if not any(np.isclose(float(v), float(u), rtol=0.0, atol=1e-12) for u in unique):
            unique.append(v)
    return tuple(unique)


def _build_analysis_config(base_cls: type[Any], overrides: dict[str, Any]) -> type[Any]:
    class AnalysisConfig(base_cls):
        pass
    for key, value in overrides.items():
        setattr(AnalysisConfig, key, value)
    return AnalysisConfig


def _build_score_weight_overrides(
    base_config: type[Any], short_weight: Any
) -> dict[str, float]:
    sw = float(short_weight)
    if not 0.0 < sw < 1.0:
        raise ValueError(f"score_short_weight must be in (0, 1), got {sw}")
    bm = float(getattr(base_config, "GATE_SCORE_MEDIUM_WEIGHT"))
    bl = float(getattr(base_config, "GATE_SCORE_LONG_WEIGHT"))
    rem = bm + bl
    if rem <= 0.0:
        raise ValueError("GATE_SCORE_MEDIUM_WEIGHT + GATE_SCORE_LONG_WEIGHT must be positive")
    r = 1.0 - sw
    return {
        "GATE_SCORE_SHORT_WEIGHT": float(sw),
        "GATE_SCORE_MEDIUM_WEIGHT": float(r * bm / rem),
        "GATE_SCORE_LONG_WEIGHT": float(r * bl / rem),
    }


def _build_param_overrides(
    base_config: type[Any], spec_name: str, value: Any
) -> dict[str, Any]:
    spec = SPEC_BY_NAME[spec_name]
    value = _coerce_value(spec, value)
    if spec.name == "score_short_weight":
        return _build_score_weight_overrides(base_config, value)
    return {spec.config_attr: value}


def _merge_overrides(*groups: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for g in groups:
        merged.update(g)
    return merged


def _load_prewarm_features(config: type[Any]) -> np.ndarray:
    cols = list(config.FEATURE_COLUMNS)
    train_df = pd.read_csv(config.get_train_data_path())
    val_df = pd.read_csv(config.get_val_data_path())
    for label, df in [("train", train_df), ("val", val_df)]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{label} set missing gate feature columns: {missing}")
    merged = pd.concat([train_df[cols], val_df[cols]], ignore_index=True)
    return merged.to_numpy(dtype=np.float32)


def _iter_complete_rounds(
    data: pd.DataFrame, round_length_steps: int
) -> list[tuple[int, int, int, pd.DataFrame]]:
    rls = int(round_length_steps)
    if rls <= 0:
        raise ValueError(f"round_length_steps must be positive, got {rls}")
    n = len(data) // rls
    return [
        (
            i,
            i * rls,
            (i + 1) * rls,
            data.iloc[i * rls : (i + 1) * rls].reset_index(drop=True),
        )
        for i in range(n)
    ]


def _prewarm_gate(gate: Any, prewarm_features: np.ndarray) -> None:
    for sample in prewarm_features:
        gate.predict(sample)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

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
    fixed_reward = float(fixed_baseline_summary.get("total_reward", 0.0))
    fixed_energy_daily = float(fixed_baseline_summary.get("E_daily_kwh_per_day", 0.0))

    signed_reward_gap_ratio = (fixed_reward - total_reward) / max(abs(fixed_reward), 1e-8)
    signed_energy_change_ratio = (energy_daily - fixed_energy_daily) / max(abs(fixed_energy_daily), 1e-8)
    event_trigger_rate = (
        float(step_results["gate_signal"].eq(1).mean()) if not step_results.empty else 0.0
    )

    return {
        "total_reward": total_reward,
        "avg_reward_per_action": float(summary.get("avg_reward_per_action", 0.0)),
        "avg_reward_per_env_step": float(summary.get("avg_reward_per_env_step", 0.0)),
        "avg_energy_score": float(summary.get("avg_energy_score", 0.0)),
        "avg_comfort_score": float(summary.get("avg_comfort_score", 0.0)),
        "action_count": int(summary.get("action_count", 0)),
        "action_frequency": float(summary.get("action_frequency", 0.0)),
        "N_daily_count_per_day": float(summary.get("N_daily_count_per_day", 0.0)),
        "event_trigger_count": int(summary.get("event_trigger_count", 0)),
        "event_trigger_rate": float(
            summary.get("event_trigger_rate", event_trigger_rate)
        ),
        "E_daily_kwh_per_day": energy_daily,
        "delta_total_reward_vs_default_gate": float(total_reward - default_reward),
        "delta_action_frequency_vs_default_gate": float(
            float(summary.get("action_frequency", 0.0)) - default_action_frequency
        ),
        "delta_N_daily_count_per_day_vs_default_gate": float(
            float(summary.get("N_daily_count_per_day", 0.0))
            - float(default_baseline_summary.get("N_daily_count_per_day", 0.0))
        ),
        "delta_event_trigger_rate_vs_default_gate": float(
            float(summary.get("event_trigger_rate", event_trigger_rate))
            - default_event_trigger_rate
        ),
        "delta_E_daily_kwh_per_day_vs_default_gate": float(energy_daily - default_energy_daily),
        "signed_reward_gap_ratio_vs_fixed_baseline": float(signed_reward_gap_ratio),
        "signed_energy_change_ratio_vs_fixed_baseline": float(signed_energy_change_ratio),
    }


def _summarize_round_metrics(
    round_rows: list[dict[str, Any]],
    metric_columns: tuple[str, ...],
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    if not round_rows:
        return {"round_count": 0}
    frame = pd.DataFrame(round_rows)
    summary: dict[str, float | int] = {"round_count": int(len(frame))}
    cv = float(NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0))
    for col in metric_columns:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").dropna()
        if vals.empty:
            continue
        m = float(vals.mean())
        s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        margin = cv * s / float(np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        summary[col] = m
        summary[f"{col}_std"] = s
        summary[f"{col}_min"] = float(vals.min())
        summary[f"{col}_max"] = float(vals.max())
        summary[f"{col}_ci_low"] = m - margin
        summary[f"{col}_ci_high"] = m + margin
    return summary


def _round_metric_row(
    summary: dict,
    step_results: pd.DataFrame,
    default_baseline_summary: dict,
    fixed_baseline_summary: dict,
    *,
    round_index: int,
    round_start_step: int,
    round_end_step: int,
    round_length_steps: int,
) -> dict[str, float | int]:
    m = _metric_summary(
        summary=summary,
        step_results=step_results,
        default_baseline_summary=default_baseline_summary,
        fixed_baseline_summary=fixed_baseline_summary,
    )
    return {
        "round_index": int(round_index + 1),
        "round_start_step": int(round_start_step + 1),
        "round_end_step": int(round_end_step),
        "round_length_steps": int(round_length_steps),
        **m,
    }


def _series_from_step_results(step_results: pd.DataFrame) -> dict[str, np.ndarray]:
    if step_results.empty:
        return {
            k: np.asarray([], dtype=np.float64)
            for k in ("reward", "action_updated", "gate_signal", "daily_energy")
        }
    dt_h = SAMPLE_INTERVAL_MINUTES / 60.0
    dur = max(float(len(step_results)) * dt_h / 24.0, 1e-12)
    return {
        "reward": step_results["reward"].to_numpy(dtype=np.float64),
        "action_updated": step_results["action_updated"].to_numpy(dtype=np.float64),
        "gate_signal": step_results["gate_signal"].to_numpy(dtype=np.float64),
        "daily_energy": step_results["power_chiller"].to_numpy(dtype=np.float64) * dt_h / dur,
    }


def _paired_metric_deltas(cand: pd.DataFrame, base: pd.DataFrame) -> dict[str, np.ndarray]:
    cs = _series_from_step_results(cand)
    bs = _series_from_step_results(base)
    n = min(len(next(iter(cs.values()))), len(next(iter(bs.values()))))
    if n <= 0:
        return {k: np.asarray([], dtype=np.float64) for k in ("total_reward", "avg_reward_per_env_step", "action_frequency", "event_trigger_rate", "E_daily_kwh_per_day")}
    rd = cs["reward"][:n] - bs["reward"][:n]
    return {
        "total_reward": rd,
        "avg_reward_per_env_step": rd,
        "action_frequency": cs["action_updated"][:n] - bs["action_updated"][:n],
        "event_trigger_rate": cs["gate_signal"][:n] - bs["gate_signal"][:n],
        "E_daily_kwh_per_day": cs["daily_energy"][:n] - bs["daily_energy"][:n],
    }


def _bootstrap_statistic(diff: np.ndarray, name: str, idx: np.ndarray) -> float:
    if diff.size == 0 or idx.size == 0:
        return 0.0
    if name in {"avg_reward_per_env_step", "action_frequency", "event_trigger_rate"}:
        return float(np.mean(diff[idx]))
    return float(np.sum(diff[idx]))


def _block_bootstrap_indices(rng: np.random.Generator, n: int, bs: int) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=np.int64)
    bs = max(1, min(bs, n))
    starts = rng.integers(0, n, size=int(np.ceil(n / bs)))
    blocks = [(s + np.arange(bs)) % n for s in starts]
    return np.concatenate(blocks).astype(np.int64)[:n]


def _paired_block_bootstrap_significance(
    cand: pd.DataFrame,
    base: pd.DataFrame,
    bootstrap_samples: int,
    block_size: int,
    seed: int,
    confidence_level: float,
    alpha: float,
) -> dict[str, float | int]:
    deltas = _paired_metric_deltas(cand, base)
    ns = len(next(iter(deltas.values()))) if deltas else 0
    names = ("total_reward", "avg_reward_per_env_step", "action_frequency", "event_trigger_rate", "E_daily_kwh_per_day")
    res: dict[str, float | int] = {
        "bootstrap_paired_steps": int(ns),
        "bootstrap_samples": int(max(0, bootstrap_samples)),
        "bootstrap_block_size": int(max(1, block_size)),
        "bootstrap_confidence_level": float(confidence_level),
        "bootstrap_significance_level": float(alpha),
    }
    if ns <= 0 or bootstrap_samples <= 0:
        for name in names:
            for suffix in ("delta_mean", "delta_ci_lower", "delta_ci_upper", "delta_p_value", "delta_significant"):
                key = f"bootstrap_{name}_{suffix}"
                res[key] = 0.0 if suffix != "delta_p_value" else 1.0
        return res

    rng = np.random.default_rng(seed)
    low = (1.0 - confidence_level) / 2.0
    high = 1.0 - low
    for name in names:
        diff = deltas[name]
        obs = _bootstrap_statistic(diff, name, np.arange(ns, dtype=np.int64))
        boot = np.asarray(
            [_bootstrap_statistic(diff, name, _block_bootstrap_indices(rng, ns, block_size)) for _ in range(bootstrap_samples)],
            dtype=np.float64,
        )
        centered = boot - obs
        p = float(np.mean(np.abs(centered) >= abs(obs)))
        cl = float(np.quantile(boot, low))
        ch = float(np.quantile(boot, high))
        res[f"bootstrap_{name}_delta_mean"] = float(obs)
        res[f"bootstrap_{name}_delta_ci_lower"] = cl
        res[f"bootstrap_{name}_delta_ci_upper"] = ch
        res[f"bootstrap_{name}_delta_p_value"] = float(min(1.0, max(0.0, p)))
        res[f"bootstrap_{name}_delta_significant"] = int((cl > 0.0 or ch < 0.0) and p < alpha)
    return res


def _gate_trace_diagnostics(cand: pd.DataFrame, base: pd.DataFrame) -> dict[str, float | int]:
    n = min(len(cand), len(base))
    diag: dict[str, float | int] = {"diagnostic_paired_steps": int(n)}
    if n <= 0:
        return {**diag, "gate_signal_diff_count": 0, "gate_signal_same_rate": 1.0, "action_update_diff_count": 0, "action_update_same_rate": 1.0, "action_value_diff_count": 0, "action_value_same_rate": 1.0}
    for col, pfx in (("gate_signal", "gate_signal"), ("action_updated", "action_update"), ("action_value", "action_value")):
        cv = cand[col].to_numpy(dtype=np.float64)[:n] if col in cand.columns else np.full(n, 0.0, dtype=np.float64)
        bv = base[col].to_numpy(dtype=np.float64)[:n] if col in base.columns else np.full(n, 0.0, dtype=np.float64)
        dc = int(np.sum(~np.isclose(cv, bv, atol=ZERO_SENSITIVITY_ABS_TOL)))
        diag[f"{pfx}_diff_count"] = dc
        diag[f"{pfx}_same_rate"] = float(1.0 - dc / n)
    return diag


def _build_candidate_gate(
    config: type[Any], test_data: pd.DataFrame, prewarm_features: np.ndarray
) -> Any:
    gate = create_streaming_gate(config=config, test_data=test_data, logger=_SilentLogger(), gate_state_path=None)
    _prewarm_gate(gate, prewarm_features)
    return gate


def _evaluate_with_gate(
    *,
    config: type[Any],
    agent: Any,
    action_space: np.ndarray,
    reward_calc: Any,
    test_data: pd.DataFrame,
    gate: Any,
) -> tuple[dict, pd.DataFrame]:
    env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
    return test_event_driven(
        agent=agent, env=env, data=test_data, action_space=action_space,
        gate=gate, feature_columns=config.FEATURE_COLUMNS, supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata builders
# ═══════════════════════════════════════════════════════════════════════════════

def _candidate_label(md: dict[str, Any]) -> str:
    if md["design_type"] == "pairwise":
        return f"{md['x_param']}={md['x_value']}, {md['y_param']}={md['y_value']}"
    return f"{md['parameter']}={md['candidate_value']}"


def _make_pairwise_metadata(
    base_config: type[Any], design: PairwiseDesign, x: Any, y: Any, idx: int
) -> dict[str, Any]:
    xs, ys = SPEC_BY_NAME[design.x_param], SPEC_BY_NAME[design.y_param]
    return {
        "design_type": "pairwise", "design": design.name, "design_label": design.label,
        "candidate_index": int(idx), "candidate_id": f"{design.name}_{idx:03d}",
        "x_param": design.x_param, "x_param_label": xs.label,
        "y_param": design.y_param, "y_param_label": ys.label,
        "x_value": _coerce_value(xs, x), "y_value": _coerce_value(ys, y),
        "x_is_baseline": int(_is_baseline_value(base_config, xs, x)),
        "y_is_baseline": int(_is_baseline_value(base_config, ys, y)),
        "parameter": "", "parameter_label": "",
        "candidate_value": np.nan, "candidate_value_numeric": np.nan,
        "is_baseline_value": int(
            _is_baseline_value(base_config, xs, x) and _is_baseline_value(base_config, ys, y)
        ),
    }


def _make_single_metadata(
    base_config: type[Any], design: SingleSweepDesign, value: Any, idx: int
) -> dict[str, Any]:
    spec = SPEC_BY_NAME[design.parameter]
    value = _coerce_value(spec, value)
    return {
        "design_type": "single", "design": design.name, "design_label": design.label,
        "candidate_index": int(idx), "candidate_id": f"{design.name}_{idx:03d}",
        "x_param": "", "x_param_label": "", "y_param": "", "y_param_label": "",
        "x_value": np.nan, "y_value": np.nan, "x_is_baseline": 0, "y_is_baseline": 0,
        "parameter": design.parameter, "parameter_label": spec.label,
        "candidate_value": value, "candidate_value_numeric": float(value),
        "is_baseline_value": int(_is_baseline_value(base_config, spec, value)),
    }


def _summarize_design_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for (_, design_name), group in frame.groupby(["design_type", "design"], sort=False):
        rv = pd.to_numeric(group["total_reward"], errors="coerce")
        nv = pd.to_numeric(group["N_daily_count_per_day"], errors="coerce")
        ev = pd.to_numeric(group["E_daily_kwh_per_day"], errors="coerce")
        sig_tr = pd.to_numeric(group.get("bootstrap_total_reward_delta_significant", 0), errors="coerce").fillna(0).sum()
        sig_ed = pd.to_numeric(group.get("bootstrap_E_daily_kwh_per_day_delta_significant", 0), errors="coerce").fillna(0).sum()
        out.append({
            "design_name": str(design_name),
            "design_label": str(group.iloc[0].get("design_label", design_name)),
            "candidate_count": int(len(group)),
            "reward_span": float(rv.max() - rv.min()),
            "N_daily_span": float(nv.max() - nv.min()),
            "E_daily_span": float(ev.max() - ev.min()),
            "significant_total_reward_count": int(sig_tr),
            "significant_E_daily_count": int(sig_ed),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_gate_mixed_sensitivity_analysis(
    train_experiment_dir: Path | None = None,
    output_dir: Path | None = None,
    data_split: str = "test",
    bootstrap_samples: int | None = None,
    bootstrap_block_size: int | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    cfg = load_config("control_compare")
    train_dir = resolve_train_experiment_dir(train_experiment_dir)
    bs_samp = int(bootstrap_samples if bootstrap_samples is not None else getattr(cfg, "GATE_SENSITIVITY_BOOTSTRAP_SAMPLES", 500))
    bs_blk = int(bootstrap_block_size if bootstrap_block_size is not None else getattr(cfg, "GATE_SENSITIVITY_BOOTSTRAP_BLOCK_SIZE", 48))
    bs_seed = int(bootstrap_seed if bootstrap_seed is not None else getattr(cfg, "GATE_SENSITIVITY_BOOTSTRAP_SEED", 42))
    cl = float(np.clip(getattr(cfg, "GATE_SENSITIVITY_CONFIDENCE_LEVEL", 0.95), 1e-6, 1.0 - 1e-6))
    sl = float(np.clip(getattr(cfg, "GATE_SENSITIVITY_SIGNIFICANCE_LEVEL", 0.05), 1e-6, 1.0 - 1e-6))
    rls = int(getattr(cfg, "GATE_SENSITIVITY_ROUND_LENGTH_STEPS", 288))

    root = Path(output_dir or (cfg.LOG_ROOT_DIR / "online_anomaly_detection" / "sensitivity")) / cfg.TIMESTAMP
    root.mkdir(parents=True, exist_ok=True)
    ctx = create_experiment_context(experiment_dir=root, config=cfg, save_config=False,
                                    log_filename=cfg.EVALUATION_LOG_FILENAME,
                                    metrics_filename=cfg.EVALUATION_METRICS_FILENAME,
                                    config_filename=cfg.CONFIG_FILENAME)
    log = ctx.logger
    log.info("Gate mixed sensitivity analysis")
    log.info(f"  train_dir={train_dir}  data_split={data_split}  output={root}")

    test_data, action_space, agent, ckpt, reward_calc = build_test_components(
        config=cfg, resolved_train_dir=train_dir, data_split=data_split,
    )
    pf = _load_prewarm_features(cfg)
    rounds = _iter_complete_rounds(test_data, rls)
    if not rounds:
        raise ValueError(f"{data_split} data length {len(test_data)} < round length {rls}")

    # Baseline gate
    bc_cfg = _build_analysis_config(cfg, {})
    bc_gate = _build_candidate_gate(bc_cfg, test_data, pf)
    bc_sum, bc_steps = _evaluate_with_gate(config=bc_cfg, agent=agent, action_space=action_space,
                                            reward_calc=reward_calc, test_data=test_data, gate=copy.deepcopy(bc_gate))
    # Fixed-step baseline
    bc_env = SequenceEnv(test_data, cfg.STATE_COLUMNS, reward_calc)
    fb_sum, fb_steps = test_fixed_interval(agent=agent, env=bc_env, action_space=action_space,
                                            fixed_interval=int(cfg.GATE_OPT_BASELINE_FIXED_INTERVAL),
                                            supply_temp_ref=cfg.CHILLER_SUPPLY_TEMP_REF,
                                            comfort_lower_bound=cfg.COMFORT_LOWER_BOUND,
                                            comfort_upper_bound=cfg.COMFORT_UPPER_BOUND)
    bc_met = _metric_summary(bc_sum, bc_steps, bc_sum, fb_sum)

    # Round baselines
    bc_rounds: list[dict[str, Any]] = []
    fb_rounds: list[dict[str, Any]] = []
    for ri, rs, re, rd in rounds:
        rrc = _create_reward_calculator(cfg, rd, action_space)
        rs_sum, rs_steps = _evaluate_with_gate(config=bc_cfg, agent=agent, action_space=action_space,
                                                reward_calc=rrc, test_data=rd, gate=copy.deepcopy(bc_gate))
        fenv = SequenceEnv(rd, cfg.STATE_COLUMNS, rrc)
        f_sum, _ = test_fixed_interval(agent=agent, env=fenv, action_space=action_space,
                                        fixed_interval=int(cfg.GATE_OPT_BASELINE_FIXED_INTERVAL),
                                        supply_temp_ref=cfg.CHILLER_SUPPLY_TEMP_REF,
                                        comfort_lower_bound=cfg.COMFORT_LOWER_BOUND, comfort_upper_bound=cfg.COMFORT_UPPER_BOUND)
        bc_rounds.append(_round_metric_row(summary=rs_sum, step_results=rs_steps,
                                            default_baseline_summary=rs_sum, fixed_baseline_summary=f_sum,
                                            round_index=ri, round_start_step=rs, round_end_step=re, round_length_steps=rls))
        fb_rounds.append({"round_index": ri+1, **_metric_summary(f_sum, pd.DataFrame(), f_sum, f_sum)})

    # ── Evaluate all candidate configurations ──────────────────────────────
    all_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []

    def _run_one(md: dict[str, Any], cand_cfg: type[Any]) -> None:
        cg = _build_candidate_gate(cand_cfg, test_data, pf)
        s, sr = _evaluate_with_gate(config=cand_cfg, agent=agent, action_space=action_space,
                                     reward_calc=reward_calc, test_data=test_data, gate=copy.deepcopy(cg))
        full_m = _metric_summary(s, sr, bc_sum, fb_sum)
        sig = _paired_block_bootstrap_significance(sr, bc_steps, bs_samp, bs_blk, bs_seed, cl, sl)
        diag = _gate_trace_diagnostics(sr, bc_steps)
        round_data_rows: list[dict[str, Any]] = []
        for ri, rs, re, rd in rounds:
            rrc = _create_reward_calculator(cand_cfg, rd, action_space)
            rs_s, rs_sr = _evaluate_with_gate(config=cand_cfg, agent=agent, action_space=action_space,
                                               reward_calc=rrc, test_data=rd, gate=copy.deepcopy(cg))
            rr = _round_metric_row(summary=rs_s, step_results=rs_sr,
                                    default_baseline_summary=bc_rounds[ri], fixed_baseline_summary=fb_rounds[ri],
                                    round_index=ri, round_start_step=rs, round_end_step=re, round_length_steps=rls)
            round_data_rows.append({**md, "candidate_label": _candidate_label(md), **rr})
        rm = _summarize_round_metrics(round_data_rows, ROUND_METRIC_COLUMNS, confidence_level=cl)
        all_rows.append({**md, "candidate_label": _candidate_label(md),
                         "full_horizon_total_reward": full_m["total_reward"],
                         "full_horizon_N_daily_count_per_day": full_m["N_daily_count_per_day"],
                         "full_horizon_E_daily_kwh_per_day": full_m["E_daily_kwh_per_day"],
                         **rm, **sig, **diag})
        round_rows.extend(round_data_rows)

    # Pairwise designs
    for design in PAIRWISE_DESIGNS:
        x_vals = _values_with_baseline(cfg, design.x_param, design.x_values)
        y_vals = _values_with_baseline(cfg, design.y_param, design.y_values)
        total = len(x_vals) * len(y_vals)
        log.info(f"Pairwise [{design.name}]: {total} combos ({len(x_vals)} x {len(y_vals)})")
        idx = 0
        for xv in x_vals:
            for yv in y_vals:
                idx += 1
                o = _merge_overrides(_build_param_overrides(cfg, design.x_param, xv),
                                      _build_param_overrides(cfg, design.y_param, yv))
                md = _make_pairwise_metadata(cfg, design, xv, yv, idx)
                _run_one(md, _build_analysis_config(cfg, o))
                log.info(f"  [{idx}/{total}] {_candidate_label(md)}  "
                         f"R={all_rows[-1]['total_reward']:.4f}  "
                         f"N={all_rows[-1]['N_daily_count_per_day']:.2f}  "
                         f"E={all_rows[-1]['E_daily_kwh_per_day']:.2f}")

    # Single-factor sweeps
    for design in SINGLE_SWEEPS:
        vals = _values_with_baseline(cfg, design.parameter, design.values)
        log.info(f"Single [{design.name}]: {len(vals)} points")
        for idx, v in enumerate(vals, 1):
            o = _build_param_overrides(cfg, design.parameter, v)
            md = _make_single_metadata(cfg, design, v, idx)
            _run_one(md, _build_analysis_config(cfg, o))
            log.info(f"  [{idx}/{len(vals)}] {_candidate_label(md)}  "
                     f"R={all_rows[-1]['total_reward']:.4f}  "
                     f"N={all_rows[-1]['N_daily_count_per_day']:.2f}  "
                     f"E={all_rows[-1]['E_daily_kwh_per_day']:.2f}")

    # ── Save results ───────────────────────────────────────────────────────
    summary_rows = _summarize_design_rows(all_rows)
    all_df = pd.DataFrame(all_rows)
    round_df = pd.DataFrame(round_rows)
    summary_df = pd.DataFrame(summary_rows)

    all_path = root / "gate_mixed_sensitivity_results.csv"
    round_path = root / "gate_mixed_sensitivity_round_results.csv"
    summary_path = root / "gate_mixed_sensitivity_summary.csv"
    baseline_path = root / "gate_mixed_sensitivity_baseline.json"

    all_df.to_csv(all_path, index=False)
    round_df.to_csv(round_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "train_experiment_dir": str(train_dir),
        "data_split": data_split,
        "pairwise_designs": [asdict(d) for d in PAIRWISE_DESIGNS],
        "single_sweeps": [asdict(d) for d in SINGLE_SWEEPS],
        "bootstrap": {"samples": bs_samp, "block_size": bs_blk, "seed": bs_seed, "confidence_level": cl, "significance_level": sl},
        "baseline": bc_met,
        "round": {"round_length_steps": rls, "round_count": len(rounds), "discarded_tail_steps": int(len(test_data) - len(rounds) * rls)},
        "best_epoch_from_train": int(ckpt.get("epoch", -1)) + 1,
        "results_csv": str(all_path),
        "round_results_csv": str(round_path),
        "summary_csv": str(summary_path),
    }
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    ctx.metrics_recorder.save_metrics(payload)

    log.info(f"Results saved: {all_path}")
    log.info(f"Round results saved: {round_path}")
    log.info(f"Summary saved: {summary_path}")
    log.info(f"Baseline meta saved: {baseline_path}")
    return payload
