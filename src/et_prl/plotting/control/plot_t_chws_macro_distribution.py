"""Generate Figure 5-2-4: macro windowed distribution of T_chws states.

Panel content:
- (a) TTC-RL-1/ST-ETC merged windowed stacked distribution
- (b) ET-PRL windowed stacked distribution

Usage:
    uv run python src/plotting/plot_t_chws_macro_distribution.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from et_prl.config.base import project_root
from et_prl.plotting._style import apply_paper_style

apply_paper_style(unicode_minus=False, tick_direction="in")


REQUIRED_STRATEGY_FILES = {
    "ET-PRL": "event_driven_step_results.csv",
    "TTC-RL-1": "fixed_interval_1_step_results.csv",
    "ST-ETC": "event_triggered_etc_step_results.csv",
}


def load_control_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load required strategy CSVs for macro distribution plotting."""
    from et_prl.plotting.control.results_path import resolve_results_dir

    results_dir = resolve_results_dir(results_dir)
    results: dict[str, pd.DataFrame] = {}
    for name, filename in REQUIRED_STRATEGY_FILES.items():
        file_path = results_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file for {name}: {file_path}")
        df = pd.read_csv(file_path)
        results[name] = df
        print(f"Loaded {name}: {len(df)} steps")

    return results


def _load_series(
    results: dict[str, pd.DataFrame], strategy_name: str
) -> tuple[np.ndarray, np.ndarray]:
    df = results[strategy_name]
    steps = df["step"].to_numpy(dtype=float)
    values = df["action_value"].to_numpy(dtype=float)
    return steps, values


def _build_temperature_color_map(
    temp_levels: np.ndarray,
) -> dict[float, tuple[float, float, float, float]]:
    """Build a low-saturation cool-to-warm color map keyed by temperature."""
    anchors: list[tuple[float, str]] = [
        (6.0, "#5B7FA6"),
        (8.0, "#8DB2D9"),
        (9.0, "#C9C9C9"),
        (15.0, "#D65F5F"),
    ]

    def hex_to_rgb01(hex_color: str) -> np.ndarray:
        hex_color = hex_color.lstrip("#")
        return np.array(
            [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)],
            dtype=float,
        )

    anchor_vals = np.array([item[0] for item in anchors], dtype=float)
    anchor_rgbs = np.array([hex_to_rgb01(item[1]) for item in anchors], dtype=float)

    color_map: dict[float, tuple[float, float, float, float]] = {}
    for level in np.asarray(temp_levels, dtype=float):
        idx = np.where(np.isclose(anchor_vals, level, rtol=0.0, atol=1e-9))[0]
        if idx.size > 0:
            rgb = anchor_rgbs[int(idx[0])]
        elif level <= anchor_vals[0]:
            rgb = anchor_rgbs[0]
        elif level >= anchor_vals[-1]:
            rgb = anchor_rgbs[-1]
        else:
            right = int(np.searchsorted(anchor_vals, level, side="right"))
            left = right - 1
            ratio = (level - anchor_vals[left]) / (anchor_vals[right] - anchor_vals[left])
            rgb = anchor_rgbs[left] * (1.0 - ratio) + anchor_rgbs[right] * ratio

        color_map[float(level)] = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    return color_map


def _compute_windowed_proportions(
    time_hours: np.ndarray,
    values: np.ndarray,
    temp_levels: np.ndarray,
    window_hours: float,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-window state proportions for stacked-percentage plotting."""
    length = min(len(time_hours), len(values))
    if length == 0 or x_max <= x_min:
        return np.asarray([], dtype=float), np.zeros((0, len(temp_levels)), dtype=float)

    t = time_hours[:length]
    v = values[:length]
    mask = (t >= x_min) & (t <= x_max)
    t = t[mask]
    v = v[mask]

    if t.size == 0:
        return np.asarray([], dtype=float), np.zeros((0, len(temp_levels)), dtype=float)

    edges = np.arange(x_min, x_max + window_hours, window_hours, dtype=float)
    if edges.size < 2:
        edges = np.asarray([x_min, x_max], dtype=float)
    elif edges[-1] < x_max:
        edges = np.append(edges, x_max)

    bin_idx = np.digitize(t, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

    proportions = np.zeros((len(edges) - 1, len(temp_levels)), dtype=float)
    counts_per_window = np.bincount(bin_idx, minlength=len(edges) - 1)

    for window in range(len(edges) - 1):
        if counts_per_window[window] == 0:
            continue
        in_window = v[bin_idx == window]
        for level_idx, level in enumerate(temp_levels):
            proportions[window, level_idx] = float(
                np.mean(np.isclose(in_window, level, rtol=0.0, atol=1e-9))
            )

    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = counts_per_window > 0
    return centers[valid], proportions[valid]


def _plot_windowed_distribution_panel(
    fig: Figure,
    panel_spec,
    row_series: list[tuple[str, np.ndarray, np.ndarray]],
    temp_levels: np.ndarray,
    temp_color_map: dict[float, tuple[float, float, float, float]],
    x_min: float,
    x_max: float,
    window_hours: float,
) -> list[Axes]:
    """Render one multi-row windowed stacked-percentage panel."""
    subgrid = panel_spec.subgridspec(len(row_series), 1, hspace=0.22)
    axes: list[Axes] = []

    for idx, (row_name, time_hours, values) in enumerate(row_series):
        ax = fig.add_subplot(subgrid[idx], sharex=axes[0] if axes else None)
        axes.append(ax)

        centers, proportions = _compute_windowed_proportions(
            time_hours=time_hours,
            values=values,
            temp_levels=temp_levels,
            window_hours=window_hours,
            x_min=x_min,
            x_max=x_max,
        )

        if centers.size > 0:
            cumulative = np.zeros(len(centers), dtype=float)
            bar_width = window_hours * 0.85
            for level_idx, level in enumerate(temp_levels):
                val = proportions[:, level_idx]
                ax.bar(
                    centers,
                    val,
                    bottom=cumulative,
                    width=bar_width,
                    color=temp_color_map[float(level)],
                    edgecolor="none",
                    align="center",
                )
                cumulative += val

        x_pad = window_hours / 2.0
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=7)
        # Keep only baseline guides at 0 and 1; remove the 0.5 dashed guide line.
        ax.grid(False)
        ax.axhline(0.0, color="#B8B8B8", linestyle="--", linewidth=0.55, alpha=0.45, zorder=0)
        ax.axhline(1.0, color="#B8B8B8", linestyle="--", linewidth=0.55, alpha=0.45, zorder=0)
        # Place subfigure title on the top-left border of each panel.
        ax.text(
            0.0,
            1.01,
            row_name,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        if idx < len(row_series) - 1:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        else:
            ax.set_xlabel("Time (h)", fontsize=7)

    return axes


def generate_macro_figure(
    results: dict[str, pd.DataFrame],
    output_path: Path,
    sampling_interval_min: float = 5.0,
    window_hours: float = 10.0,
) -> None:
    """Generate and save Figure 5-2-4 macro distribution."""
    required = ("ET-PRL", "TTC-RL-1", "ST-ETC")
    missing = [name for name in required if name not in results]
    if missing:
        raise KeyError(f"Missing required strategies: {', '.join(missing)}")

    et_steps, et_values = _load_series(results, "ET-PRL")
    ttc1_steps, ttc1_values = _load_series(results, "TTC-RL-1")
    st_steps, st_values = _load_series(results, "ST-ETC")

    common_length = min(len(ttc1_values), len(st_values))
    if common_length > 0:
        equal_mask = np.isclose(
            ttc1_values[:common_length], st_values[:common_length], rtol=0.0, atol=1e-9
        )
        equal_ratio = float(np.mean(equal_mask))
        different_count = int(common_length - int(np.sum(equal_mask)))
        print(
            f"TTC-RL-1 vs ST-ETC action equality ratio: {equal_ratio:.4f} "
            f"(different steps: {different_count}/{common_length})"
        )

    time_et = et_steps * sampling_interval_min / 60.0
    time_ttc1 = ttc1_steps * sampling_interval_min / 60.0
    time_st = st_steps * sampling_interval_min / 60.0

    merged_time = np.concatenate([time_ttc1, time_st])
    merged_values = np.concatenate([ttc1_values, st_values])
    merge_order = np.argsort(merged_time, kind="mergesort")
    merged_time = merged_time[merge_order]
    merged_values = merged_values[merge_order]

    series_map: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "(a) TTC-RL-1/ST-ETC": (merged_time, merged_values),
        "(b) ET-PRL": (time_et, et_values),
    }
    all_values = [vals for _, vals in series_map.values()]
    temp_levels = np.unique(np.concatenate(all_values))
    temp_color_map = _build_temperature_color_map(temp_levels)

    preferred_order = ["(a) TTC-RL-1/ST-ETC", "(b) ET-PRL"]
    panel_rows = [
        (name, series_map[name][0], series_map[name][1])
        for name in preferred_order
        if name in series_map
    ]

    fig_height = 4.28
    fig = plt.figure(figsize=(7.06, fig_height))
    outer = fig.add_gridspec(1, 1, left=0.07, right=0.98, top=0.90, bottom=0.12)
    panel_axes = _plot_windowed_distribution_panel(
        fig=fig,
        panel_spec=outer[0],
        row_series=panel_rows,
        temp_levels=temp_levels,
        temp_color_map=temp_color_map,
        x_min=0.0,
        x_max=350.0,
        window_hours=window_hours,
    )

    fig.supylabel(r"Proportion of $T_{\mathrm{chws}}$ Setpoints", fontsize=6, x=0.015)

    temp_handles = [
        Patch(
            facecolor=temp_color_map[float(level)],
            edgecolor="none",
            label=f"{float(level):g} °C",
        )
        for level in temp_levels
    ]
    panel_axes[0].legend(
        handles=temp_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=min(4, len(temp_handles)),
        fontsize=5.5,
        framealpha=0.94,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.05,
        bbox_transform=fig.transFigure,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg", dpi=100, pad_inches=0.02)
    plt.close(fig)
    print(f"Saved Figure 5-2-3 to {output_path}")


def main(results_dir: Path | None = None, output_dir: Path | None = None) -> None:
    if results_dir is None:
        from et_prl.plotting.control.results_path import default_control_compare_results_dir

        results_dir = default_control_compare_results_dir()
    if output_dir is None:
        output_dir = project_root() / "outputs" / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading control results from {results_dir}...")
    results = load_control_results(results_dir)
    output_path = output_dir / "fig6_t_chws_macro_distribution.svg"
    generate_macro_figure(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 5-2-3 macro T_chws distribution.")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    main(results_dir=args.results_dir, output_dir=args.output_dir)
