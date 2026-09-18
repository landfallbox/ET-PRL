"""门控敏感性分析的指标与统计计算。

本模块只含纯计算函数（轮次指标汇总、配对块自助法显著性、门控轨迹诊断等），
不依赖实验编排或 I/O。常量取自 sensitivity_design。
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from et_prl.experiments.sensitivity_design import (
    SAMPLE_INTERVAL_MINUTES,
    ZERO_SENSITIVITY_ABS_TOL,
)

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
    round_rows: list[dict],
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
