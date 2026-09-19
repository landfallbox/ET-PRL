"""Generate Figure 5-2-6: typical-day load-trigger alignment.

This figure provides direct temporal evidence for whether trigger updates
concentrate around rapidly changing load periods.

Usage:
    uv run python src/plotting/plot_trigger_load_alignment.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from et_prl.config.base import project_root
from et_prl.plotting._style import apply_paper_style

apply_paper_style(unicode_minus=True)


COLOR_LOAD = "#0072B2"
COLOR_ET = "#C43C2B"
COLOR_ST = "#1F5AA6"
COLOR_TTC = "#9AA0A6"
LOAD_LINEWIDTH = 1.2
LOAD_ALPHA = 0.90


def _load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    from et_prl.plotting.control.results_path import resolve_results_dir

    results_dir = resolve_results_dir(results_dir)
    mapping = {
        "ET-PRL": "event_driven_step_results.csv",
        "ST-ETC": "event_triggered_etc_step_results.csv",
        "TTC-RL-1": "fixed_interval_1_step_results.csv",
    }
    results: dict[str, pd.DataFrame] = {}
    for name, file_name in mapping.items():
        path = results_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing required result file: {path}")
        results[name] = pd.read_csv(path)
    return results


def _select_typical_day_by_range(load_series: np.ndarray, steps_per_day: int) -> int:
    n_days = int(np.ceil(len(load_series) / steps_per_day))
    day_ranges: list[tuple[int, float]] = []
    for day in range(1, n_days + 1):
        start = (day - 1) * steps_per_day
        end = min(day * steps_per_day, len(load_series))
        if end - start < max(8, steps_per_day // 4):
            continue
        local = load_series[start:end]
        day_ranges.append((day, float(np.max(local) - np.min(local))))

    if not day_ranges:
        return 1

    ranges = np.asarray([item[1] for item in day_ranges], dtype=float)
    target = float(np.median(ranges))
    selected_day = min(day_ranges, key=lambda x: abs(x[1] - target))[0]
    return int(selected_day)


def _style_axes(ax: Axes, tick_size: float = 10.5) -> None:
    ax.tick_params(axis="both", which="both", labelsize=tick_size, top=False, right=False)
    ax.tick_params(axis="y", which="both", labelright=False, right=False)
    ax.yaxis.set_ticks_position("left")
    for spine in ax.spines.values():
        spine.set_linewidth(0.95)


def generate_trigger_alignment_figure(
    results_dir: Path,
    env_data_path: Path,
    output_path: Path,
    sampling_interval_min: float = 5.0,
    selected_day: int | None = None,
    high_change_quantile: float = 0.85,
) -> None:
    results = _load_results(results_dir)
    if not env_data_path.exists():
        raise FileNotFoundError(f"Environment data not found: {env_data_path}")

    env_df = pd.read_csv(env_data_path)
    if "CL" not in env_df.columns:
        raise KeyError("Expected CL column in environment data for cooling load series.")

    lengths = [len(df) for df in results.values()]
    n = min(min(lengths), len(env_df))
    if n <= 0:
        raise ValueError("No aligned samples available for plotting.")

    steps_per_day = int(round(24 * 60 / sampling_interval_min))
    load_series = env_df["CL"].to_numpy(dtype=float)[:n]
    load_delta_abs = np.abs(np.diff(load_series, prepend=load_series[0]))
    high_thr = float(np.quantile(load_delta_abs, high_change_quantile))

    if selected_day is None:
        selected_day = _select_typical_day_by_range(load_series, steps_per_day)

    start = (selected_day - 1) * steps_per_day
    end = min(selected_day * steps_per_day, n)
    if end - start < max(8, steps_per_day // 4):
        raise ValueError(f"Selected day {selected_day} has insufficient samples.")

    local_idx = np.arange(start, end, dtype=int)
    local_t = (local_idx - start) * sampling_interval_min / 60.0
    local_load = load_series[start:end]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.15], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.84, bottom=0.16, hspace=0.05)

    ax_top.set_axisbelow(True)
    ax_bottom.set_axisbelow(True)

    ax_top.plot(
        local_t, local_load, color=COLOR_LOAD, linewidth=LOAD_LINEWIDTH, alpha=LOAD_ALPHA, zorder=2
    )

    ax_top.set_ylabel("Cooling load (kW)", fontsize=11.5, labelpad=8.5)
    ax_top.grid(axis="y", color="#D6DCE5", linewidth=0.7, linestyle="--", alpha=0.30)

    top_legend_handles = [
        Line2D(
            [0], [0], color=COLOR_LOAD, lw=LOAD_LINEWIDTH, alpha=LOAD_ALPHA, label="Cooling load"
        ),
    ]
    ax_top.legend(
        handles=top_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=1,
        frameon=False,
        fontsize=9.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    _style_axes(ax_top, tick_size=10.5)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

    strategy_rows = [
        ("TTC-RL-1", 0.70, COLOR_TTC),
        ("ST-ETC", 1.35, COLOR_ST),
        ("ET-PRL", 2.00, COLOR_ET),
    ]

    for name, y_level, color in strategy_rows:
        df = results[name].iloc[:n]
        updated = (
            pd.to_numeric(df["action_updated"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            > 0.5
        )
        local_updated = updated[start:end]
        trigger_t = local_t[local_updated]
        if len(trigger_t) > 0:
            ax_bottom.vlines(
                trigger_t,
                y_level - 0.24,
                y_level + 0.24,
                color=color,
                alpha=0.70,
                linewidth=0.75,
                zorder=2,
            )

    ax_bottom.set_yticks([0.70, 1.35, 2.00])
    ax_bottom.set_yticklabels(["TTC-RL-1", "ST-ETC", "ET-PRL"], fontsize=9.5)
    ax_bottom.set_ylim(0.30, 2.35)
    ax_bottom.set_xlim(0.0, 24.0)
    ax_bottom.set_ylabel("Trigger policies", fontsize=11.0, labelpad=10.0)
    ax_bottom.grid(axis="x", color="#D6DCE5", linewidth=0.7, linestyle="--", alpha=0.30)
    ax_bottom.grid(axis="y", color="#DCE2EA", linewidth=0.6, linestyle="--", alpha=0.30)
    _style_axes(ax_bottom, tick_size=10.5)

    major_xticks = [0, 6, 12, 18, 24]
    major_xticklabels = ["00:00", "06:00", "12:00", "18:00", "24:00"]
    ax_top.set_xticks(major_xticks)
    ax_bottom.set_xticks(major_xticks)
    ax_top.xaxis.set_minor_locator(MultipleLocator(3))
    ax_bottom.xaxis.set_minor_locator(MultipleLocator(3))
    ax_top.yaxis.set_major_locator(MultipleLocator(1000))
    ax_top.yaxis.set_minor_locator(MultipleLocator(500))
    ax_top.tick_params(axis="y", which="minor", length=2.5)
    ax_bottom.tick_params(axis="x", which="minor", length=2.5)
    ax_bottom.set_xticklabels(major_xticklabels, fontsize=10.5)
    fig.supxlabel("Time of day (h)", fontsize=11.5, y=0.065)
    fig.align_ylabels([ax_top, ax_bottom])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")
    print(f"Selected day: {selected_day}")
    print(f"High-change threshold |ΔQ_load| (q={high_change_quantile:.2f}): {high_thr:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 5-2-6 trigger-load alignment.")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--env-data-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--sampling-interval-min", type=float, default=5.0)
    parser.add_argument("--selected-day", type=int, default=None)
    parser.add_argument("--high-change-quantile", type=float, default=0.85)
    args = parser.parse_args()

    root = project_root()
    if args.results_dir is None:
        from et_prl.plotting.control.results_path import default_control_compare_results_dir

        args.results_dir = default_control_compare_results_dir()
    results_dir = args.results_dir
    env_data_path = args.env_data_path or (root / "data" / "dqn" / "test_data.csv")
    output_path = args.output_path or (
        root / "outputs" / "figures" / "fig9_trigger_load_alignment.svg"
    )

    generate_trigger_alignment_figure(
        results_dir=results_dir,
        env_data_path=env_data_path,
        output_path=output_path,
        sampling_interval_min=args.sampling_interval_min,
        selected_day=args.selected_day,
        high_change_quantile=args.high_change_quantile,
    )


if __name__ == "__main__":
    main()
