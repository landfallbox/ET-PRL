"""Generate the latest gate parameter sensitivity figure from sensitivity CSV results.

Usage:
    uv run python src/plotting/plot_latest_gate_parameter_sensitivity.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
    }
)


DEFAULT_RESULTS_RELATIVE_PATH = Path(
    "logs/online_anomaly_detection/sensitivity/20260524_162519/gate_parameter_sensitivity_results.csv"
)
DEFAULT_OUTPUT_RELATIVE_PATH = Path("docs/pics/fig12_gate_key_parameter_sensitivity.svg")


@dataclass(frozen=True)
class ParameterPanel:
    name: str
    label: str
    panel_label: str


SELECTED_PANELS = [
    ParameterPanel("threshold_bias", "Threshold bias", "a"),
    ParameterPanel("trigger_hysteresis_margin", "Hysteresis margin", "b"),
    ParameterPanel("local_window_size", "Local window size", "c"),
    ParameterPanel("threshold_quantile", "Threshold quantile", "d"),
    ParameterPanel("score_short_weight", "Short scale weight", "e"),
    ParameterPanel("threshold_mad_scale", "MAD scale", "f"),
]


REQUIRED_COLUMNS = {
    "parameter",
    "candidate_value_numeric",
    "is_baseline_value",
    "total_reward",
    "action_count",
    "E_daily_kwh_per_day",
}


def _load_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"Sensitivity results file not found: {results_path}")

    results = pd.read_csv(results_path)
    missing = REQUIRED_COLUMNS.difference(results.columns)
    if missing:
        raise KeyError(f"Missing required columns in {results_path}: {sorted(missing)}")

    for column in ["candidate_value_numeric", "total_reward", "action_count", "E_daily_kwh_per_day"]:
        results[column] = pd.to_numeric(results[column], errors="coerce")
    results = results.dropna(subset=["candidate_value_numeric", "total_reward", "action_count", "E_daily_kwh_per_day"])
    if results.empty:
        raise ValueError(f"No valid sensitivity rows found in: {results_path}")
    return results


def _baseline_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["is_baseline_value"].astype(str).str.lower().isin({"true", "1", "yes"})


def _plot_panel(ax: plt.Axes, frame: pd.DataFrame, panel: ParameterPanel, norm: colors.Normalize) -> None:
    data = frame[frame["parameter"] == panel.name].sort_values("candidate_value_numeric").copy()
    if data.empty:
        raise ValueError(f"No rows found for parameter: {panel.name}")

    baseline_rows = data[_baseline_mask(data)]
    if baseline_rows.empty:
        raise ValueError(f"No baseline row found for parameter: {panel.name}")

    baseline = baseline_rows.iloc[0]
    best = data.loc[data["total_reward"].idxmax()]
    reward_span = float(data["total_reward"].max() - data["total_reward"].min())

    size = 42.0 + 2.5 * data["action_count"].clip(lower=0.0)
    scatter = ax.scatter(
        data["candidate_value_numeric"],
        data["total_reward"],
        s=size,
        c=data["E_daily_kwh_per_day"],
        cmap="viridis",
        norm=norm,
        alpha=0.86,
        edgecolor="#1F2937",
        linewidth=0.55,
        zorder=3,
    )
    ax.plot(
        data["candidate_value_numeric"],
        data["total_reward"],
        color="#6B7280",
        linewidth=1.0,
        alpha=0.72,
        zorder=2,
    )
    ax.axhline(float(baseline["total_reward"]), color="#374151", linestyle="--", linewidth=1.0, zorder=1)
    ax.scatter(
        [float(baseline["candidate_value_numeric"])],
        [float(baseline["total_reward"])],
        s=92,
        marker="D",
        color="#F97316",
        edgecolor="#111827",
        linewidth=0.75,
        zorder=5,
    )
    ax.scatter(
        [float(best["candidate_value_numeric"])],
        [float(best["total_reward"])],
        s=135,
        marker="*",
        color="#DC2626",
        edgecolor="#111827",
        linewidth=0.75,
        zorder=6,
    )

    ax.set_title(f"{panel.panel_label}  {panel.label}", loc="left", fontsize=10.8, fontweight="bold")
    ax.set_xlabel("Candidate value", fontsize=9.6)
    ax.set_ylabel("Total reward", fontsize=9.6)
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.25)
    ax.tick_params(axis="both", labelsize=8.8)
    ax.text(
        0.03,
        0.94,
        f"Reward span {reward_span:.1f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.86},
    )
    return scatter


def generate_figure(results_path: Path, output_path: Path) -> None:
    results = _load_results(results_path)
    selected_results = results[results["parameter"].isin(panel.name for panel in SELECTED_PANELS)]
    norm = colors.Normalize(
        vmin=float(selected_results["E_daily_kwh_per_day"].min()),
        vmax=float(selected_results["E_daily_kwh_per_day"].max()),
    )

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.5), dpi=150, constrained_layout=True)
    scatters = []
    for ax, panel in zip(axes.flatten(), SELECTED_PANELS, strict=True):
        scatters.append(_plot_panel(ax, results, panel, norm))

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3B82F6", markeredgecolor="#1F2937", markersize=6.5, label="Candidate"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#F97316", markeredgecolor="#111827", markersize=6.5, label="Default gate"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#DC2626", markeredgecolor="#111827", markersize=9, label="Best reward"),
        Line2D([0], [0], color="#374151", linestyle="--", linewidth=1.0, label="Default reward"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9.0,
        bbox_to_anchor=(0.5, 1.02),
    )
    colorbar = fig.colorbar(scatters[-1], ax=axes, shrink=0.82, pad=0.015)
    colorbar.set_label("Daily energy kWh per day", fontsize=9.2)
    colorbar.ax.tick_params(labelsize=8.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved latest gate sensitivity figure: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the latest gate parameter sensitivity figure.")
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    generate_figure(
        results_path=args.results_path or root / DEFAULT_RESULTS_RELATIVE_PATH,
        output_path=args.output_path or root / DEFAULT_OUTPUT_RELATIVE_PATH,
    )


if __name__ == "__main__":
    main()
