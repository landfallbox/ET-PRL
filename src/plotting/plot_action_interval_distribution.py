"""Generate Figure 5-2-5: action update interval distribution.

This script redraws the interval distribution with interpretable bins instead of
collapsing all long intervals into one bucket.

Default bins (steps):
- 1
- 2
- 3
- 4-6
- 7-15
- 16-30
- 31-74

Usage:
    uv run python src/plotting/plot_action_interval_distribution.py
    uv run python src/plotting/plot_action_interval_distribution.py --y-scale log
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import LogLocator, MultipleLocator


plt.rcParams.update(
    {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'axes.unicode_minus': True,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    }
)


STRATEGY_FILES = {
    'TTC-RL-1': 'fixed_interval_1_step_results.csv',
    'ST-ETC': 'event_triggered_etc_step_results.csv',
    'ET-PRL': 'event_driven_step_results.csv',
}

STRATEGY_ORDER = ['TTC-RL-1', 'ST-ETC', 'ET-PRL']

STRATEGY_COLORS = {
    'TTC-RL-1': '#6B7280',
    'ST-ETC': '#1F4E79',
    'ET-PRL': '#E64B35',
}

BASE_BINS: list[tuple[int, int]] = [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 6),
    (7, 15),
    (16, 30),
    (31, 74),
]


def _load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for strategy, file_name in STRATEGY_FILES.items():
        path = results_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f'Missing required result file: {path}')
        results[strategy] = pd.read_csv(path)
    return results


def _extract_trigger_intervals(df: pd.DataFrame) -> np.ndarray:
    if 'action_updated' not in df.columns or 'step' not in df.columns:
        raise KeyError("Expected 'action_updated' and 'step' columns in result file.")

    updated = pd.to_numeric(df['action_updated'], errors='coerce').fillna(0.0).to_numpy(dtype=float) > 0.5
    trigger_steps = pd.to_numeric(df.loc[updated, 'step'], errors='coerce').dropna().astype(int).to_numpy()

    if trigger_steps.size <= 1:
        return np.array([], dtype=int)

    trigger_steps = np.sort(trigger_steps)
    intervals = np.diff(trigger_steps)
    intervals = intervals[intervals > 0]
    return intervals.astype(int)


def _build_bins(intervals_by_strategy: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    max_interval = 0
    for intervals in intervals_by_strategy.values():
        if intervals.size > 0:
            max_interval = max(max_interval, int(np.max(intervals)))

    bins = list(BASE_BINS)
    if max_interval > BASE_BINS[-1][1]:
        bins.append((BASE_BINS[-1][1] + 1, max_interval))
    return bins


def _format_bin_label(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f'{start}-{end}'


def _compute_binned_share(intervals: np.ndarray, bins: list[tuple[int, int]]) -> np.ndarray:
    shares = np.zeros(len(bins), dtype=float)
    if intervals.size == 0:
        return shares

    total = float(intervals.size)
    for idx, (start, end) in enumerate(bins):
        count = int(np.sum((intervals >= start) & (intervals <= end)))
        shares[idx] = 100.0 * count / total
    return shares


def _style_axes(ax: plt.Axes, labelsize: int = 9, tick_length: float = 4.0) -> None:
    ax.tick_params(
        axis='both',
        labelsize=labelsize,
        direction='out',
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
        length=tick_length,
        width=0.8,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def _draw_grouped_bars(
    ax: plt.Axes,
    x: np.ndarray,
    labels: list[str],
    shares_by_strategy: dict[str, np.ndarray],
    y_scale: str,
    width: float,
) -> None:
    for idx, strategy in enumerate(STRATEGY_ORDER):
        values = shares_by_strategy[strategy]
        plot_values = values.copy()
        if y_scale == 'log':
            plot_values[plot_values <= 0.0] = np.nan

        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            plot_values,
            width=width,
            color=STRATEGY_COLORS[strategy],
            edgecolor='white',
            linewidth=0.7,
            alpha=0.95,
            label=strategy,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)


def _add_tail_inset(
    ax: plt.Axes,
    labels: list[str],
    shares_by_strategy: dict[str, np.ndarray],
    width: float,
) -> plt.Axes | None:
    if len(labels) <= 4:
        return None

    tail_start_idx = 3
    tail_x = np.arange(len(labels) - tail_start_idx, dtype=float)
    tail_labels = labels[tail_start_idx:]

    inset = ax.inset_axes([0.57, 0.23, 0.34, 0.53])
    for idx, strategy in enumerate(STRATEGY_ORDER):
        values = shares_by_strategy[strategy][tail_start_idx:]
        offset = (idx - 1) * width
        inset.bar(
            tail_x + offset,
            values,
            width=width,
            color=STRATEGY_COLORS[strategy],
            edgecolor='white',
            linewidth=0.6,
            alpha=0.95,
            zorder=3,
        )

    max_tail = 0.0
    for strategy in STRATEGY_ORDER:
        strategy_tail = shares_by_strategy[strategy][tail_start_idx:]
        if strategy_tail.size > 0:
            max_tail = max(max_tail, float(np.max(strategy_tail)))

    if max_tail <= 0.0:
        max_tail = 1.0

    inset.set_xticks(tail_x)
    inset.set_xticklabels(tail_labels, fontsize=9)
    inset.set_ylim(0.0, max_tail * 1.32)
    inset.yaxis.set_major_locator(MultipleLocator(0.5 if max_tail <= 2.0 else 1.0))
    inset.set_ylabel('Share (%)', fontsize=10, labelpad=8)
    inset.minorticks_off()
    inset.grid(axis='y', color='#EFEFEF', linewidth=0.5, zorder=0)
    _style_axes(inset, labelsize=9, tick_length=3.0)

    for spine in inset.spines.values():
        spine.set_linewidth(0.8)

    return inset


def _add_inset_guidance_box(
    ax: plt.Axes,
    inset: plt.Axes,
    x: np.ndarray,
    shares_by_strategy: dict[str, np.ndarray],
    tail_start_idx: int,
    width: float,
) -> None:
    if x.size <= tail_start_idx:
        return

    tail_max = 0.0
    for strategy in STRATEGY_ORDER:
        values = shares_by_strategy[strategy][tail_start_idx:]
        if values.size > 0:
            tail_max = max(tail_max, float(np.max(values)))

    box_ymax = max(2.4, tail_max * 1.18)
    half_group_span = 1.5 * width
    box_padding = 0.04
    box_xmin = float(x[tail_start_idx] - half_group_span - box_padding)
    box_xmax = float(x[-1] + half_group_span + box_padding)

    rect = Rectangle(
        (box_xmin, 0.0),
        box_xmax - box_xmin,
        box_ymax,
        fill=False,
        linestyle='--',
        linewidth=0.9,
        edgecolor='#9CA3AF',
        zorder=4,
    )
    ax.add_patch(rect)

    left_conn = ConnectionPatch(
        xyA=(box_xmin, box_ymax),
        coordsA=ax.transData,
        xyB=(0.00, 0.00),
        coordsB=inset.transAxes,
        linestyle='--',
        linewidth=0.8,
        color='#9CA3AF',
        zorder=4,
    )
    right_conn = ConnectionPatch(
        xyA=(box_xmax, box_ymax),
        coordsA=ax.transData,
        xyB=(1.00, 0.00),
        coordsB=inset.transAxes,
        linestyle='--',
        linewidth=0.8,
        color='#9CA3AF',
        zorder=4,
    )
    ax.add_artist(left_conn)
    ax.add_artist(right_conn)


def generate_interval_distribution_figure(
    results_dir: Path,
    output_path: Path,
    y_scale: str = 'linear',
    sampling_interval_min: float = 5.0,
) -> None:
    if y_scale not in {'linear', 'log'}:
        raise ValueError("y_scale must be one of: 'linear', 'log'.")

    results = _load_results(results_dir)
    intervals_by_strategy = {name: _extract_trigger_intervals(df) for name, df in results.items()}

    bins = _build_bins(intervals_by_strategy)
    labels = [_format_bin_label(start, end) for start, end in bins]

    shares_by_strategy = {
        name: _compute_binned_share(intervals=intervals_by_strategy[name], bins=bins) for name in STRATEGY_ORDER
    }

    x = np.arange(len(labels), dtype=float)
    width = 0.23

    fig, ax = plt.subplots(figsize=(9.3, 6.2), constrained_layout=True)
    _draw_grouped_bars(
        ax=ax,
        x=x,
        labels=labels,
        shares_by_strategy=shares_by_strategy,
        y_scale=y_scale,
        width=width,
    )

    if y_scale == 'linear':
        ax.set_ylim(0.0, 103.0)
        ax.yaxis.set_major_locator(MultipleLocator(20.0))
        ax.minorticks_off()
        inset = _add_tail_inset(ax=ax, labels=labels, shares_by_strategy=shares_by_strategy, width=width)
        if inset is not None:
            _add_inset_guidance_box(
                ax=ax,
                inset=inset,
                x=x,
                shares_by_strategy=shares_by_strategy,
                tail_start_idx=3,
                width=width,
            )
    else:
        positive_values = []
        for strategy in STRATEGY_ORDER:
            vals = shares_by_strategy[strategy]
            positive_values.extend(vals[vals > 0.0].tolist())

        if not positive_values:
            raise ValueError('No positive shares available for log scale plotting.')

        min_positive = float(np.min(np.asarray(positive_values, dtype=float)))
        lower = max(0.01, min_positive * 0.70)
        ax.set_yscale('log')
        ax.set_ylim(lower, 120.0)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax.minorticks_off()

    ax.set_xlabel('Trigger interval categories (steps)', fontsize=13)
    ax.set_ylabel('Share (%)', fontsize=13)
    ax.grid(axis='y', which='major', color='#EAEAEA', linewidth=0.8, zorder=0)
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(0.985, 0.995),
        ncol=1,
        fontsize=11,
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.0,
    )
    _style_axes(ax, labelsize=11, tick_length=4.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format='svg', dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved figure: {output_path}')
    print(f'Binning labels: {labels}')

    for strategy in STRATEGY_ORDER:
        intervals = intervals_by_strategy[strategy]
        max_interval = int(np.max(intervals)) if intervals.size > 0 else 0
        max_hours = max_interval * sampling_interval_min / 60.0
        values = shares_by_strategy[strategy]
        rounded = ', '.join(f'{val:.2f}%' for val in values)
        print(f'- {strategy}: n={intervals.size}, max_interval={max_interval} steps (~{max_hours:.2f} h), shares=[{rounded}]')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Figure 5-2-5 action update interval distribution.')
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--output-path', type=Path, default=None)
    parser.add_argument('--y-scale', type=str, choices=['linear', 'log'], default='linear')
    parser.add_argument('--sampling-interval-min', type=float, default=5.0)
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    results_dir = args.results_dir or (root / 'logs' / 'control_compare' / '20260402_222140')
    output_path = args.output_path or (root / 'docs' / 'pics' / 'fig8_action_update_interval_distribution.svg')

    generate_interval_distribution_figure(
        results_dir=results_dir,
        output_path=output_path,
        y_scale=args.y_scale,
        sampling_interval_min=args.sampling_interval_min,
    )


if __name__ == '__main__':
    main()
