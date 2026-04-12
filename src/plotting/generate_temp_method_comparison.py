"""Generate a temporary comparison figure for six control-trajectory visualizations.

The figure compares ET/TTC/ST strategy trajectories with:
1) Discrete scatter + faceting
2) Horizontal segment timeline
3) Windowed stacked percentage bars
4) Run-duration boxplot
5) Sankey-style state transition links
6) Transition matrix heatmaps

Usage:
    uv run python src/plotting/generate_temp_method_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


STRATEGY_FILES: dict[str, str] = {
    "ET": "event_driven_step_results.csv",
    "TTC": "fixed_interval_1_step_results.csv",
    "ST": "event_triggered_etc_step_results.csv",
}
STRATEGY_ORDER: list[str] = ["ET", "TTC", "ST"]

SAMPLE_INTERVAL_HOURS = 5.0 / 60.0
DEFAULT_STATE_LEVELS = np.asarray([6.0, 8.0, 9.0, 15.0], dtype=float)


def _build_state_color_map(state_levels: np.ndarray) -> dict[float, tuple[float, float, float, float]]:
    base_map: dict[float, tuple[float, float, float, float]] = {
        6.0: (0.133, 0.369, 0.659, 1.0),
        8.0: (0.255, 0.714, 0.769, 1.0),
        9.0: (0.631, 0.855, 0.705, 1.0),
        15.0: (0.941, 0.231, 0.125, 1.0),
    }

    missing_levels = [float(level) for level in state_levels if float(level) not in base_map]
    if missing_levels:
        cmap = plt.get_cmap("viridis")
        for idx, level in enumerate(sorted(missing_levels)):
            base_map[level] = cmap((idx + 1) / (len(missing_levels) + 1))
    return base_map


def _resolve_results_dir(explicit_results_dir: Path | None, control_compare_root: Path) -> Path:
    if explicit_results_dir is not None:
        if not explicit_results_dir.exists():
            raise FileNotFoundError(f"results_dir does not exist: {explicit_results_dir}")
        return explicit_results_dir

    if not control_compare_root.exists():
        raise FileNotFoundError(f"Control-compare root not found: {control_compare_root}")

    candidates = sorted(path for path in control_compare_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No control-compare experiment directories under: {control_compare_root}")
    return candidates[-1]


def _load_series(results_dir: Path) -> dict[str, pd.DataFrame]:
    series: dict[str, pd.DataFrame] = {}
    for strategy, filename in STRATEGY_FILES.items():
        file_path = results_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file for {strategy}: {file_path}")

        df = pd.read_csv(file_path)
        required_columns = {"step", "action_value"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"{file_path} does not contain required columns: {required_columns}")

        steps = df["step"].to_numpy(dtype=float)
        values = df["action_value"].to_numpy(dtype=float)
        order = np.argsort(steps)
        steps = steps[order]
        values = values[order]

        time_h = (steps - float(np.min(steps))) * SAMPLE_INTERVAL_HOURS
        series[strategy] = pd.DataFrame(
            {
                "time_h": time_h,
                "state": values,
            }
        )
    return series


def _infer_state_levels(series: dict[str, pd.DataFrame]) -> np.ndarray:
    observed = np.concatenate([df["state"].to_numpy(dtype=float) for df in series.values()])
    merged = np.unique(np.concatenate([DEFAULT_STATE_LEVELS, observed]))
    return np.asarray(sorted(float(value) for value in merged), dtype=float)


def _build_segments(time_h: np.ndarray, states: np.ndarray) -> list[tuple[float, float, float]]:
    length = min(len(time_h), len(states))
    if length == 0:
        return []

    t = time_h[:length]
    s = states[:length]
    segments: list[tuple[float, float, float]] = []

    start = float(t[0])
    current = float(s[0])
    for idx in range(1, length):
        if not np.isclose(s[idx], current, rtol=0.0, atol=1e-9):
            end = float(t[idx])
            if end > start:
                segments.append((start, end, current))
            start = end
            current = float(s[idx])

    end = float(t[-1] + SAMPLE_INTERVAL_HOURS)
    if end > start:
        segments.append((start, end, current))
    return segments


def _compute_window_proportions(
    time_h: np.ndarray,
    states: np.ndarray,
    state_levels: np.ndarray,
    window_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(time_h), len(states))
    if length == 0:
        return np.asarray([], dtype=float), np.zeros((0, len(state_levels)), dtype=float)

    t = time_h[:length]
    s = states[:length]
    end_time = float(t[-1] + SAMPLE_INTERVAL_HOURS)

    edges = np.arange(0.0, end_time + window_hours, window_hours, dtype=float)
    if len(edges) < 2:
        edges = np.asarray([0.0, max(window_hours, end_time)], dtype=float)
    elif edges[-1] < end_time:
        edges = np.append(edges, edges[-1] + window_hours)

    bin_idx = np.digitize(t, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

    proportions = np.zeros((len(edges) - 1, len(state_levels)), dtype=float)
    counts_per_window = np.bincount(bin_idx, minlength=len(edges) - 1)

    for window in range(len(edges) - 1):
        if counts_per_window[window] == 0:
            continue
        in_window = s[bin_idx == window]
        for level_idx, level in enumerate(state_levels):
            proportions[window, level_idx] = float(np.mean(np.isclose(in_window, level, rtol=0.0, atol=1e-9)))

    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = counts_per_window > 0
    return centers[valid], proportions[valid]


def _build_run_duration_dataframe(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        df = series[strategy]
        segments = _build_segments(df["time_h"].to_numpy(dtype=float), df["state"].to_numpy(dtype=float))
        for start, end, state in segments:
            rows.append(
                {
                    "strategy": strategy,
                    "state": float(state),
                    "duration_h": float(end - start),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["strategy", "state", "duration_h"])
    return pd.DataFrame(rows)


def _build_transition_matrix(states: np.ndarray, state_levels: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(state_levels), len(state_levels)), dtype=int)
    if len(states) < 2:
        return matrix

    index_map = {round(float(level), 6): idx for idx, level in enumerate(state_levels)}
    for current, nxt in zip(states[:-1], states[1:]):
        src = index_map.get(round(float(current), 6))
        dst = index_map.get(round(float(nxt), 6))
        if src is None or dst is None:
            continue
        matrix[src, dst] += 1
    return matrix


def _panel_scatter(
    fig: plt.Figure,
    panel_spec,
    series: dict[str, pd.DataFrame],
    state_levels: np.ndarray,
    x_max: float,
    strategy_colors: dict[str, str],
) -> None:
    subgrid = GridSpecFromSubplotSpec(3, 1, subplot_spec=panel_spec, hspace=0.08)
    shared_ax = None

    for idx, strategy in enumerate(STRATEGY_ORDER):
        ax = fig.add_subplot(subgrid[idx, 0], sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax

        df = series[strategy]
        ax.scatter(
            df["time_h"],
            df["state"],
            s=6,
            alpha=0.70,
            color=strategy_colors[strategy],
            edgecolors="none",
        )
        ax.set_ylim(float(np.min(state_levels) - 0.6), float(np.max(state_levels) + 0.6))
        ax.set_xlim(0.0, x_max)
        ax.set_yticks(state_levels)
        ax.set_ylabel(strategy, fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

        if idx < len(STRATEGY_ORDER) - 1:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        else:
            ax.set_xlabel("Time (h)", fontsize=10)

        if idx == 0:
            ax.set_title("A. Discrete Scatter + Faceting", loc="left", fontsize=12, fontweight="bold")


def _panel_horizontal_segments(
    fig: plt.Figure,
    panel_spec,
    series: dict[str, pd.DataFrame],
    state_levels: np.ndarray,
    x_max: float,
    state_colors: dict[float, tuple[float, float, float, float]],
) -> None:
    subgrid = GridSpecFromSubplotSpec(3, 1, subplot_spec=panel_spec, hspace=0.10)
    shared_ax = None

    for idx, strategy in enumerate(STRATEGY_ORDER):
        ax = fig.add_subplot(subgrid[idx, 0], sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax

        df = series[strategy]
        segments = _build_segments(df["time_h"].to_numpy(dtype=float), df["state"].to_numpy(dtype=float))
        for start, end, state in segments:
            ax.hlines(
                y=state,
                xmin=start,
                xmax=end,
                colors=[state_colors[float(state)]],
                linewidth=3.2,
                alpha=0.95,
            )

        ax.set_ylim(float(np.min(state_levels) - 0.8), float(np.max(state_levels) + 0.8))
        ax.set_xlim(0.0, x_max)
        ax.set_yticks(state_levels)
        ax.set_ylabel(strategy, fontsize=9)
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)

        if idx < len(STRATEGY_ORDER) - 1:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        else:
            ax.set_xlabel("Time (h)", fontsize=10)

        if idx == 0:
            ax.set_title("B. Horizontal Segment Timeline", loc="left", fontsize=12, fontweight="bold")


def _panel_windowed_stacked(
    fig: plt.Figure,
    panel_spec,
    series: dict[str, pd.DataFrame],
    state_levels: np.ndarray,
    x_max: float,
    state_colors: dict[float, tuple[float, float, float, float]],
    window_hours: float,
) -> None:
    subgrid = GridSpecFromSubplotSpec(3, 1, subplot_spec=panel_spec, hspace=0.10)
    shared_ax = None

    for idx, strategy in enumerate(STRATEGY_ORDER):
        ax = fig.add_subplot(subgrid[idx, 0], sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax

        df = series[strategy]
        centers, proportions = _compute_window_proportions(
            df["time_h"].to_numpy(dtype=float),
            df["state"].to_numpy(dtype=float),
            state_levels=state_levels,
            window_hours=window_hours,
        )

        if len(centers) > 0:
            cumulative = np.zeros(len(centers), dtype=float)
            for level_idx, level in enumerate(state_levels):
                values = proportions[:, level_idx]
                ax.bar(
                    centers,
                    values,
                    bottom=cumulative,
                    width=window_hours * 0.85,
                    color=state_colors[float(level)],
                    edgecolor="none",
                    align="center",
                )
                cumulative += values

        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.0, x_max)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_ylabel(strategy, fontsize=9)
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)

        if idx < len(STRATEGY_ORDER) - 1:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        else:
            ax.set_xlabel("Time (h)", fontsize=10)

        if idx == 0:
            ax.set_title(
                "C. Windowed Stacked Percentage (Macro Trend)",
                loc="left",
                fontsize=12,
                fontweight="bold",
            )


def _panel_duration_boxplot(
    ax: plt.Axes,
    duration_df: pd.DataFrame,
    state_levels: np.ndarray,
    state_colors: dict[float, tuple[float, float, float, float]],
) -> None:
    state_display = sorted(float(level) for level in state_levels)
    positions: list[float] = []
    samples: list[np.ndarray] = []
    box_colors: list[tuple[float, float, float, float]] = []

    group_span = len(state_display) + 1
    for strategy_idx, strategy in enumerate(STRATEGY_ORDER):
        strategy_rows = duration_df[duration_df["strategy"] == strategy]
        for state_idx, state in enumerate(state_display):
            vals = strategy_rows.loc[np.isclose(strategy_rows["state"].to_numpy(dtype=float), state), "duration_h"].to_numpy(
                dtype=float
            )
            if vals.size == 0:
                continue

            positions.append(float(strategy_idx * group_span + state_idx + 1))
            samples.append(vals)
            box_colors.append(state_colors[float(state)])

    if samples:
        box = ax.boxplot(
            samples,
            positions=positions,
            widths=0.7,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
        )
        for patch, color in zip(box["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
    else:
        ax.text(0.5, 0.5, "No duration data", transform=ax.transAxes, ha="center", va="center")

    centers = [strategy_idx * group_span + (len(state_display) + 1) / 2.0 for strategy_idx in range(len(STRATEGY_ORDER))]
    ax.set_xticks(centers)
    ax.set_xticklabels(STRATEGY_ORDER)
    ax.set_ylabel("Run Duration (h)")
    ax.set_title("D. Run-Duration Distribution (Boxplot)", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    handles = [
        Patch(facecolor=state_colors[float(level)], edgecolor="none", label=f"{int(level)}")
        for level in sorted(float(level) for level in state_levels)
    ]
    ax.legend(handles=handles, title="State", ncol=min(4, len(handles)), loc="upper right", fontsize=8)


def _draw_sankey_like(
    ax: plt.Axes,
    matrix: np.ndarray,
    state_levels: np.ndarray,
    state_colors: dict[float, tuple[float, float, float, float]],
    subtitle: str,
    with_panel_title: bool,
) -> None:
    display_levels = sorted(float(level) for level in state_levels)[::-1]
    y_positions = np.linspace(0.88, 0.12, len(display_levels), dtype=float)
    y_map = {level: y for level, y in zip(display_levels, y_positions)}

    left_x, right_x = 0.08, 0.84
    node_width, node_height = 0.08, 0.10

    max_count = float(np.max(matrix)) if matrix.size > 0 and np.max(matrix) > 0 else 1.0
    level_to_index = {float(level): idx for idx, level in enumerate(state_levels)}

    for src_level in display_levels:
        src_idx = level_to_index[src_level]
        for dst_level in display_levels:
            dst_idx = level_to_index[dst_level]
            count = float(matrix[src_idx, dst_idx])
            if count <= 0.0:
                continue

            y0 = y_map[src_level]
            y1 = y_map[dst_level]
            verts = [
                (left_x + node_width, y0),
                (0.42, y0),
                (0.58, y1),
                (right_x, y1),
            ]
            path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
            width = 0.8 + 8.5 * (count / max_count)
            alpha = 0.18 + 0.42 * (count / max_count)
            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=state_colors[src_level],
                linewidth=width,
                alpha=alpha,
                capstyle="round",
            )
            ax.add_patch(patch)

    for level in display_levels:
        y = y_map[level]
        color = state_colors[level]
        ax.add_patch(Rectangle((left_x, y - node_height / 2.0), node_width, node_height, facecolor=color, edgecolor="none"))
        ax.add_patch(Rectangle((right_x, y - node_height / 2.0), node_width, node_height, facecolor=color, edgecolor="none"))
        ax.text(left_x - 0.015, y, f"{int(level)}", ha="right", va="center", fontsize=8)

    ax.text(left_x + node_width / 2.0, 0.99, "Current", ha="center", va="top", fontsize=8)
    ax.text(right_x + node_width / 2.0, 0.99, "Next", ha="center", va="top", fontsize=8)

    if with_panel_title:
        ax.set_title("E. State Transition Sankey-style", loc="left", fontsize=12, fontweight="bold")
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def _panel_transition_heatmaps(
    fig: plt.Figure,
    panel_spec,
    transition_matrices: dict[str, np.ndarray],
    state_levels: np.ndarray,
) -> None:
    subgrid = GridSpecFromSubplotSpec(1, 3, subplot_spec=panel_spec, wspace=0.35)

    probabilities: dict[str, np.ndarray] = {}
    for strategy, matrix in transition_matrices.items():
        row_sum = matrix.sum(axis=1, keepdims=True)
        prob = np.divide(
            matrix.astype(float),
            row_sum,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sum > 0,
        )
        probabilities[strategy] = prob

    vmax = max(float(np.max(prob)) for prob in probabilities.values())
    vmax = max(vmax, 1e-6)

    image_ref = None
    axes: list[plt.Axes] = []
    tick_labels = [str(int(level)) for level in state_levels]
    for idx, strategy in enumerate(STRATEGY_ORDER):
        ax = fig.add_subplot(subgrid[0, idx])
        axes.append(ax)
        prob = probabilities[strategy]

        image_ref = ax.imshow(prob, cmap="YlOrRd", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(state_levels)))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(range(len(state_levels)))
        if idx == 0:
            ax.set_yticklabels(tick_labels, fontsize=8)
            ax.set_ylabel("Next state")
            ax.set_title("F. Transition Matrix Heatmap\nET", fontsize=11, fontweight="bold", loc="left")
        else:
            ax.set_yticklabels([])
            ax.set_title(strategy, fontsize=10)

        if idx == 1:
            ax.set_xlabel("Current state")

        for row in range(prob.shape[0]):
            for col in range(prob.shape[1]):
                value = prob[row, col]
                text_color = "black" if value < vmax * 0.55 else "white"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    if image_ref is not None:
        colorbar = fig.colorbar(image_ref, ax=axes, fraction=0.045, pad=0.02)
        colorbar.set_label("Transition probability", fontsize=8)
        colorbar.ax.tick_params(labelsize=8)


def generate_figure(results_dir: Path, output_path: Path, window_hours: float) -> None:
    series = _load_series(results_dir)
    state_levels = _infer_state_levels(series)
    state_colors = _build_state_color_map(state_levels)
    strategy_colors = {
        "ET": "#1f77b4",
        "TTC": "#ff7f0e",
        "ST": "#2ca02c",
    }

    x_max = max(float(df["time_h"].max() + SAMPLE_INTERVAL_HOURS) for df in series.values())

    transition_matrices: dict[str, np.ndarray] = {}
    for strategy in STRATEGY_ORDER:
        transition_matrices[strategy] = _build_transition_matrix(
            series[strategy]["state"].to_numpy(dtype=float),
            state_levels=state_levels,
        )

    duration_df = _build_run_duration_dataframe(series)

    fig = plt.figure(figsize=(20, 20))
    outer = GridSpec(3, 2, figure=fig, wspace=0.20, hspace=0.28, height_ratios=[1.18, 1.0, 1.10])

    _panel_scatter(fig, outer[0, 0], series, state_levels, x_max, strategy_colors)
    _panel_horizontal_segments(fig, outer[0, 1], series, state_levels, x_max, state_colors)
    _panel_windowed_stacked(fig, outer[1, 0], series, state_levels, x_max, state_colors, window_hours=window_hours)

    box_ax = fig.add_subplot(outer[1, 1])
    _panel_duration_boxplot(box_ax, duration_df, state_levels, state_colors)

    sankey_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2, 0], wspace=0.24)
    for idx, strategy in enumerate(STRATEGY_ORDER):
        ax = fig.add_subplot(sankey_grid[0, idx])
        _draw_sankey_like(
            ax=ax,
            matrix=transition_matrices[strategy],
            state_levels=state_levels,
            state_colors=state_colors,
            subtitle=strategy,
            with_panel_title=(idx == 0),
        )

    _panel_transition_heatmaps(fig, outer[2, 1], transition_matrices, state_levels)

    legend_handles = [
        Patch(facecolor=state_colors[float(level)], edgecolor="none", label=f"{int(level)}")
        for level in sorted(float(level) for level in state_levels)
    ]
    fig.legend(
        handles=legend_handles,
        title="Temperature state",
        ncol=min(4, len(legend_handles)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )

    fig.suptitle(
        "Temporary Comparison of Control-Trajectory Visualization Methods",
        fontsize=15,
        y=0.998,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a temporary 6-method control visualization comparison figure.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing strategy step-result CSV files. Defaults to the latest logs/control_compare run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / "pics" / "temp_control_visualization_methods.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=10.0,
        help="Window size (hours) for the stacked-percentage panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path("logs") / "control_compare"
    results_dir = _resolve_results_dir(args.results_dir, root)
    print(f"Using results directory: {results_dir}")
    generate_figure(results_dir=results_dir, output_path=args.output, window_hours=float(args.window_hours))
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()