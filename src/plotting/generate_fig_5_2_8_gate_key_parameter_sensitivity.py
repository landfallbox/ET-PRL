"""Generate Figure 5-2-8: gate key parameter sensitivity and near-optimal region.

Usage:
    uv run python src/plotting/generate_fig_5_2_8_gate_key_parameter_sensitivity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
    }
)


FIGURE_FILENAME = "fig_5_2_8_gate_key_parameter_sensitivity_and_near_optimal_region.svg"
DEFAULT_NEAR_OPTIMAL_TOLERANCE = 0.10


PHASE1_PARAMS: list[tuple[str, str]] = [
    ("threshold_bias", "Threshold Bias"),
    ("threshold_quantile", "Threshold Quantile"),
    ("contamination", "Contamination"),
]

PHASE2_PARAMS: list[tuple[str, str]] = [
    ("alpha_local_weight", "Local Weight"),
    ("global_ema_decay", "Global EMA Decay"),
    ("local_window_size", "Local Window Size"),
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
    near_mask: np.ndarray,
    parameter_col: str,
    parameter_label: str,
    point_color: str,
    best_value: float,
    near_threshold: float,
    show_ylabel: bool,
) -> None:
    x_all = pd.to_numeric(trials[parameter_col], errors="coerce").to_numpy(dtype=float)
    y_all = pd.to_numeric(trials["objective_score"], errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(x_all) & np.isfinite(y_all)
    if not np.any(valid_mask):
        raise ValueError(f"No valid numeric points for parameter: {parameter_col}")

    x_all = x_all[valid_mask]
    y_all = y_all[valid_mask]
    near_mask = near_mask[valid_mask]

    x_near = x_all[near_mask]
    y_near = y_all[near_mask]
    if x_near.size == 0:
        best_idx = int(np.argmin(y_all))
        x_near = np.asarray([x_all[best_idx]], dtype=float)
        y_near = np.asarray([y_all[best_idx]], dtype=float)

    near_min = float(np.min(x_near))
    near_max = float(np.max(x_near))
    if np.isclose(near_min, near_max):
        pad = max(1e-4, 0.03 * max(1.0, abs(near_min)))
        near_min -= pad
        near_max += pad

    ax.axvspan(near_min, near_max, color=point_color, alpha=0.14, zorder=0)
    ax.scatter(x_all, y_all, s=18, color="#B8C0CC", alpha=0.65, edgecolors="none", zorder=1)
    ax.scatter(x_near, y_near, s=30, color=point_color, alpha=0.95, edgecolors="white", linewidths=0.4, zorder=3)

    ax.axhline(best_value, color="#4B5563", linestyle="--", linewidth=1.0, zorder=2)
    ax.axhline(near_threshold, color="#6B7280", linestyle=":", linewidth=0.9, zorder=2)

    ax.set_xlabel(parameter_label, fontsize=10.5)
    if show_ylabel:
        ax.set_ylabel("Objective Score (lower is better)", fontsize=10.5)

    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
    ax.tick_params(axis="both", labelsize=9.5)

    range_text = f"Near-optimal: [{_format_range_value(float(np.min(x_near)))}, {_format_range_value(float(np.max(x_near)))}]"
    ax.text(
        0.02,
        0.98,
        range_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#1F2937",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D1D5DB",
            "alpha": 0.95,
        },
    )


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

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.4), dpi=140)

    fig.suptitle(
        "Figure 5-2-8  Gate Key Parameter Sensitivity and Near-optimal Region",
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )
    fig.text(0.012, 0.932, "Phase 1: Threshold-decision parameters", fontsize=11.2, fontweight="semibold")
    fig.text(0.012, 0.468, "Phase 2: Statistical-structure parameters", fontsize=11.2, fontweight="semibold")

    phase1_color = "#2A7F62"
    phase2_color = "#1F5AA6"

    for idx, (column, label) in enumerate(PHASE1_PARAMS):
        _plot_parameter_panel(
            ax=axes[0, idx],
            trials=phase1_trials,
            near_mask=phase1_mask,
            parameter_col=column,
            parameter_label=label,
            point_color=phase1_color,
            best_value=phase1_best,
            near_threshold=phase1_threshold,
            show_ylabel=(idx == 0),
        )

    for idx, (column, label) in enumerate(PHASE2_PARAMS):
        _plot_parameter_panel(
            ax=axes[1, idx],
            trials=phase2_trials,
            near_mask=phase2_mask,
            parameter_col=column,
            parameter_label=label,
            point_color=phase2_color,
            best_value=phase2_best,
            near_threshold=phase2_threshold,
            show_ylabel=(idx == 0),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, format="svg", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 5-2-8: {output_path}")
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
