"""Generate Figure 5-2-8: gate key parameter sensitivity and near-optimal region.

Usage:
    uv run python src/plotting/plot_gate_parameter_sensitivity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter, SymmetricalLogLocator


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
    }
)


FIGURE_FILENAME = "fig12_gate_key_parameter_sensitivity.svg"
DEFAULT_NEAR_OPTIMAL_TOLERANCE = 0.10
SYMLOG_LINTHRESH = 1.0
SYMLOG_CANDIDATE_TICKS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0]
FOCUS_CLIP_QUANTILE = 0.88
FOCUS_NEAR_THRESHOLD_MULTIPLIER = 2.0
FOCUS_MIN_POINTS = 24
BEST_MARKER_COLOR = "#1F7A5A"
NON_BEST_POINT_COLOR = "#6B7280"
NON_BEST_POINT_ALPHA = 0.58
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


PHASE1_PARAMS: list[tuple[str, str]] = [
    ("threshold_bias", r"Threshold Bias ($b_{\mathrm{bias}}$)"),
    ("threshold_quantile", r"Threshold Quantile ($q_{\mathrm{thr}}$)"),
    ("threshold_mad_scale", r"MAD Scale ($\kappa_{\mathrm{mad}}$)"),
]

PHASE2_PARAMS: list[tuple[str, str]] = [
    ("alpha_local_weight", r"Local Weight ($\alpha_{\mathrm{local}}$)"),
    ("global_ema_decay", r"Global EMA Decay ($\beta_{\mathrm{ema}}$)"),
    ("local_window_size", r"Local Window Size ($W_{\mathrm{local}}$)"),
]


def _load_trials(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Trials file not found: {csv_path}")

    trials = pd.read_csv(csv_path)
    required_columns = {"state", "objective_score"}
    missing = required_columns.difference(trials.columns)
    if missing:
        raise KeyError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    trials = trials[trials["state"] == "COMPLETE"].copy()
    trials["objective_score"] = pd.to_numeric(trials["objective_score"], errors="coerce")
    trials = trials.dropna(subset=["objective_score"]).reset_index(drop=True)
    if trials.empty:
        raise ValueError(f"No complete valid trials found in: {csv_path}")
    return trials


def _ensure_parameter_columns(trials: pd.DataFrame, params: list[tuple[str, str]], phase_name: str) -> None:
    missing = [column for column, _ in params if column not in trials.columns]
    if missing:
        raise KeyError(f"Missing parameter columns in {phase_name}: {missing}")


def _compute_near_optimal_mask(
    objective_scores: pd.Series,
    tolerance: float,
) -> tuple[np.ndarray, float, float]:
    values = objective_scores.to_numpy(dtype=float)
    best_value = float(np.min(values))

    # Supports both non-negative and potentially negative objective values.
    near_optimal_threshold = best_value + abs(best_value) * tolerance
    mask = values <= (near_optimal_threshold + 1e-12)
    if not np.any(mask):
        mask[int(np.argmin(values))] = True

    return mask, best_value, near_optimal_threshold


def _format_range_value(value: float) -> str:
    if abs(value - round(value)) <= 1e-9:
        return f"{int(round(value))}"
    return f"{value:.4f}"


def _plot_parameter_panel(
    ax: plt.Axes,
    trials: pd.DataFrame,
    parameter_col: str,
    parameter_label: str,
    best_value: float,
    show_ylabel: bool,
) -> None:
    x_all = pd.to_numeric(trials[parameter_col], errors="coerce").to_numpy(dtype=float)
    y_all = pd.to_numeric(trials["objective_score"], errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(x_all) & np.isfinite(y_all)
    if not np.any(valid_mask):
        raise ValueError(f"No valid numeric points for parameter: {parameter_col}")

    x_all = x_all[valid_mask]
    y_all = y_all[valid_mask]

    ax.scatter(
        x_all,
        y_all,
        s=14,
        color=NON_BEST_POINT_COLOR,
        alpha=NON_BEST_POINT_ALPHA,
        edgecolors="none",
        zorder=1,
    )

    best_mask = np.isclose(
        y_all,
        best_value,
        rtol=0.0,
        atol=max(1e-12, 1e-8 * max(1.0, abs(best_value))),
    )
    if not np.any(best_mask):
        best_mask[int(np.argmin(y_all))] = True

    ax.scatter(
        x_all[best_mask],
        y_all[best_mask],
        s=14,
        marker="o",
        color=BEST_MARKER_COLOR,
        edgecolors="#111827",
        linewidths=0.75,
        zorder=5,
    )

    ax.axhline(best_value, color="#4B5563", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=1.15)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(
            base=10,
            linthresh=SYMLOG_LINTHRESH,
            subs=np.arange(2, 10),
        )
    )
    ax.yaxis.set_minor_formatter(NullFormatter())

    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    x_span = max(x_max - x_min, 1e-9)
    left_pad = max(0.04 * x_span, 0.01 * max(1.0, abs(x_min)))
    right_pad = max(0.07 * x_span, 0.015 * max(1.0, abs(x_max)))
    if parameter_col == "local_window_size":
        right_pad = max(right_pad, 20.0)
    ax.set_xlim(x_min - left_pad, x_max + right_pad)

    ax.set_xlabel(parameter_label, fontsize=10.5)
    if show_ylabel:
        ax.set_ylabel("Objective Score", fontsize=10.2)
    else:
        ax.tick_params(axis="y", which="both", labelleft=False)

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.18)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.tick_params(axis="y", which="minor", length=2.8, width=0.65, color="#6B7280")


def _compute_y_limits(objective_scores: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(objective_scores, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No valid objective scores for y-axis limits")

    y_min = float(np.min(values))
    y_max = float(np.max(values))
    spread = max(y_max - y_min, 1e-6)

    if y_min > 0:
        # Keep the lower bound positive to avoid showing a mathematically invalid 0 tick for log-like scaling.
        lower = max(0.1, min(0.5, 0.85 * y_min))
    else:
        lower = y_min - 0.05 * spread
    upper = y_max + 0.05 * spread
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def _apply_row_y_ticks(row_axes: np.ndarray, y_limits: tuple[float, float]) -> None:
    lower, upper = y_limits
    ticks = [tick for tick in SYMLOG_CANDIDATE_TICKS if lower <= tick <= upper * 1.001]
    if not ticks:
        return

    for ax in row_axes:
        ax.set_yticks(ticks)


def _add_panel_labels(axes: np.ndarray) -> None:
    for label, ax in zip(PANEL_LABELS, axes.flatten(), strict=False):
        ax.text(
            -0.02,
            1.045,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.8,
            fontweight="bold",
            clip_on=False,
        )


def _filter_focus_trials(
    trials: pd.DataFrame,
    near_mask: np.ndarray,
    best_value: float,
    near_threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, int, float]:
    y_all = pd.to_numeric(trials["objective_score"], errors="coerce").to_numpy(dtype=float)
    total_count = int(y_all.size)
    if total_count == 0:
        return trials.copy(), near_mask.copy(), 0, float("nan")

    q_cap = float(np.quantile(y_all, FOCUS_CLIP_QUANTILE))
    if near_threshold >= 0:
        near_cap = near_threshold * FOCUS_NEAR_THRESHOLD_MULTIPLIER
    else:
        near_cap = near_threshold + abs(near_threshold) * (FOCUS_NEAR_THRESHOLD_MULTIPLIER - 1.0)

    clip_cap = min(q_cap, near_cap)
    topk_idx = min(FOCUS_MIN_POINTS - 1, total_count - 1)
    topk_cap = float(np.partition(y_all, topk_idx)[topk_idx])
    clip_cap = max(clip_cap, topk_cap)
    clip_cap = max(clip_cap, best_value + 1e-9)

    keep_mask = (y_all <= (clip_cap + 1e-12)) | near_mask
    filtered_trials = trials.loc[keep_mask].copy().reset_index(drop=True)
    filtered_near = near_mask[keep_mask]

    removed_count = int(total_count - filtered_trials.shape[0])
    return filtered_trials, filtered_near, removed_count, clip_cap


def _print_near_optimal_ranges(
    phase_name: str,
    trials: pd.DataFrame,
    near_mask: np.ndarray,
    params: list[tuple[str, str]],
    tolerance: float,
) -> None:
    print(f"{phase_name} near-optimal ranges (tolerance={tolerance:.0%}):")
    for column, label in params:
        values = pd.to_numeric(trials.loc[near_mask, column], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        low = _format_range_value(float(np.min(values)))
        high = _format_range_value(float(np.max(values)))
        print(f"{label}: [{low}, {high}]")


def generate_figure(
    phase1_trials_path: Path,
    phase2_trials_path: Path,
    output_path: Path,
    near_optimal_tolerance: float = DEFAULT_NEAR_OPTIMAL_TOLERANCE,
) -> None:
    if near_optimal_tolerance < 0:
        raise ValueError("near_optimal_tolerance must be non-negative")

    phase1_trials = _load_trials(phase1_trials_path)
    phase2_trials = _load_trials(phase2_trials_path)
    _ensure_parameter_columns(phase1_trials, PHASE1_PARAMS, phase_name="phase1")
    _ensure_parameter_columns(phase2_trials, PHASE2_PARAMS, phase_name="phase2")

    phase1_mask, phase1_best, phase1_threshold = _compute_near_optimal_mask(
        phase1_trials["objective_score"],
        tolerance=near_optimal_tolerance,
    )
    phase2_mask, phase2_best, phase2_threshold = _compute_near_optimal_mask(
        phase2_trials["objective_score"],
        tolerance=near_optimal_tolerance,
    )

    phase1_trials_focus, _, phase1_removed, phase1_cap = _filter_focus_trials(
        trials=phase1_trials,
        near_mask=phase1_mask,
        best_value=phase1_best,
        near_threshold=phase1_threshold,
    )
    phase2_trials_focus, _, phase2_removed, phase2_cap = _filter_focus_trials(
        trials=phase2_trials,
        near_mask=phase2_mask,
        best_value=phase2_best,
        near_threshold=phase2_threshold,
    )

    phase1_y_limits = _compute_y_limits(phase1_trials_focus["objective_score"])
    phase2_y_limits = _compute_y_limits(phase2_trials_focus["objective_score"])

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.4), dpi=140, sharey="row")

    for idx, (column, label) in enumerate(PHASE1_PARAMS):
        _plot_parameter_panel(
            ax=axes[0, idx],
            trials=phase1_trials_focus,
            parameter_col=column,
            parameter_label=label,
            best_value=phase1_best,
            show_ylabel=(idx == 0),
        )

    for idx, (column, label) in enumerate(PHASE2_PARAMS):
        _plot_parameter_panel(
            ax=axes[1, idx],
            trials=phase2_trials_focus,
            parameter_col=column,
            parameter_label=label,
            best_value=phase2_best,
            show_ylabel=(idx == 0),
        )

    for ax in axes[0, :]:
        ax.set_ylim(*phase1_y_limits)
    for ax in axes[1, :]:
        ax.set_ylim(*phase2_y_limits)

    _apply_row_y_ticks(axes[0, :], phase1_y_limits)
    _apply_row_y_ticks(axes[1, :], phase2_y_limits)
    _add_panel_labels(axes)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=NON_BEST_POINT_COLOR,
            markeredgecolor="none",
            markersize=6,
            alpha=NON_BEST_POINT_ALPHA,
            label="Displayed trials",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BEST_MARKER_COLOR,
            markeredgecolor="#111827",
            markeredgewidth=0.55,
            markersize=6,
            label="Best trial(s)",
        ),
        Line2D(
            [0],
            [0],
            color="#4B5563",
            linestyle="--",
            linewidth=1.0,
            label="Best objective (baseline)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=8.6,
        handlelength=1.9,
        columnspacing=1.4,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.subplots_adjust(hspace=0.35)
    fig.savefig(output_path, format="svg", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 5-2-8: {output_path}")
    print(
        "Phase 1 focus filter: "
        f"removed {phase1_removed} high-value points, "
        f"kept {len(phase1_trials_focus)}/{len(phase1_trials)}, "
        f"display cap={phase1_cap:.4f}"
    )
    print(
        "Phase 2 focus filter: "
        f"removed {phase2_removed} high-value points, "
        f"kept {len(phase2_trials_focus)}/{len(phase2_trials)}, "
        f"display cap={phase2_cap:.4f}"
    )
    _print_near_optimal_ranges(
        phase_name="Phase 1",
        trials=phase1_trials,
        near_mask=phase1_mask,
        params=PHASE1_PARAMS,
        tolerance=near_optimal_tolerance,
    )
    _print_near_optimal_ranges(
        phase_name="Phase 2",
        trials=phase2_trials,
        near_mask=phase2_mask,
        params=PHASE2_PARAMS,
        tolerance=near_optimal_tolerance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 5-2-8 gate key parameter sensitivity and near-optimal region.",
    )
    parser.add_argument("--phase1-trials-path", type=Path, default=None)
    parser.add_argument("--phase2-trials-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--near-optimal-tolerance", type=float, default=DEFAULT_NEAR_OPTIMAL_TOLERANCE)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    phase1_default = (
        root
        / "logs"
        / "online_anomaly_detection"
        / "optimization"
        / "20260324_142434"
        / "phase1"
        / "gate"
        / "gate_trials.csv"
    )
    phase2_default = (
        root
        / "logs"
        / "online_anomaly_detection"
        / "optimization"
        / "20260324_190718"
        / "phase2"
        / "gate"
        / "gate_trials.csv"
    )

    generate_figure(
        phase1_trials_path=args.phase1_trials_path or phase1_default,
        phase2_trials_path=args.phase2_trials_path or phase2_default,
        output_path=args.output_path or (root / "docs" / "pics" / FIGURE_FILENAME),
        near_optimal_tolerance=args.near_optimal_tolerance,
    )


if __name__ == "__main__":
    main()
