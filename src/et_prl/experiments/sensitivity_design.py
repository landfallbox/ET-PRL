"""门控敏感性分析的设计定义。

本模块只含数据类与设计常量（参数规格、成对/单因子设计、轮次指标列），
不含任何分析逻辑。供 gate_sensitivity.py 与 sensitivity_stats.py 共用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from et_prl.evaluation.control.common import (
    SAMPLE_INTERVAL_MINUTES,  # noqa: F401  (供 sensitivity_stats 等复用)
)


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
    SensitivitySpec(
        "trigger_hysteresis_margin", "GATE_TRIGGER_HYSTERESIS_MARGIN", "m_hys", "boundary"
    ),
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

ZERO_SENSITIVITY_ABS_TOL = 1e-9

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
