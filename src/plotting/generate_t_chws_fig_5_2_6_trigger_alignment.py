"""Generate Figure 5-2-6: typical-day load-trigger alignment.

This figure provides direct temporal evidence for whether trigger updates
concentrate around rapidly changing load periods.

Usage:
    uv run python src/plotting/generate_t_chws_fig_5_2_6_trigger_alignment.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'axes.unicode_minus': True,
    }
)


COLOR_LOAD = '#2F3A4A'
COLOR_HIGH = '#E69F00'
COLOR_ET = '#E64B35'
COLOR_ST = '#1F4E79'
COLOR_TTC = '#6B7280'


def _load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    mapping = {
        'ET-PRL': 'event_driven_step_results.csv',
        'ST-ETC': 'event_triggered_etc_step_results.csv',
        'TTC-RL-1': 'fixed_interval_1_step_results.csv',
    }
    results: dict[str, pd.DataFrame] = {}
    for name, file_name in mapping.items():
        path = results_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f'Missing required result file: {path}')
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


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis='both', labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def generate_trigger_alignment_figure(
    results_dir: Path,
    env_data_path: Path,
    output_path: Path,
    sampling_interval_min: float = 5.0,
    selected_day: Optional[int] = None,
    high_change_quantile: float = 0.85,
) -> None:
    results = _load_results(results_dir)
    if not env_data_path.exists():
        raise FileNotFoundError(f'Environment data not found: {env_data_path}')

    env_df = pd.read_csv(env_data_path)
    if 'CL' not in env_df.columns:
        raise KeyError('Expected CL column in environment data for cooling load series.')

    lengths = [len(df) for df in results.values()]
    n = min(min(lengths), len(env_df))
    if n <= 0:
        raise ValueError('No aligned samples available for plotting.')

    steps_per_day = int(round(24 * 60 / sampling_interval_min))
    load_series = env_df['CL'].to_numpy(dtype=float)[:n]
    load_delta_abs = np.abs(np.diff(load_series, prepend=load_series[0]))
    high_thr = float(np.quantile(load_delta_abs, high_change_quantile))
    high_mask_full = load_delta_abs >= high_thr

    if selected_day is None:
        selected_day = _select_typical_day_by_range(load_series, steps_per_day)

    start = (selected_day - 1) * steps_per_day
    end = min(selected_day * steps_per_day, n)
    if end - start < max(8, steps_per_day // 4):
        raise ValueError(f'Selected day {selected_day} has insufficient samples.')

    local_idx = np.arange(start, end, dtype=int)
    local_t = (local_idx - start) * sampling_interval_min / 60.0
    local_load = load_series[start:end]
    local_high = high_mask_full[start:end]

    fig = plt.figure(figsize=(13.5, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.9, 1.1], hspace=0.10)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)

    ax_top.plot(local_t, local_load, color=COLOR_LOAD, linewidth=1.8, label='Cooling load (kW)', zorder=2)
    ax_top.scatter(
        local_t[local_high],
        local_load[local_high],
        s=14,
        color=COLOR_HIGH,
        alpha=0.85,
        label=r'High $|\Delta Q_{load}|$ (global q85)',
        zorder=3,
    )

    ax_top.set_ylabel('Cooling load (kW)', fontsize=10)
    ax_top.grid(axis='y', color='#ECECEC', linewidth=0.6)
    ax_top.legend(loc='upper right', fontsize=8, frameon=False)
    ax_top.text(0.01, 0.98, '(a)', transform=ax_top.transAxes, va='top', ha='left', fontsize=10, fontweight='bold')
    _style_axes(ax_top)
    ax_top.tick_params(axis='x', labelbottom=False)

    strategy_rows = [
        ('TTC-RL-1', 1.0, COLOR_TTC),
        ('ST-ETC', 2.0, COLOR_ST),
        ('ET-PRL', 3.0, COLOR_ET),
    ]

    for name, y_level, color in strategy_rows:
        df = results[name].iloc[:n]
        updated = pd.to_numeric(df['action_updated'], errors='coerce').fillna(0.0).to_numpy(dtype=float) > 0.5
        local_updated = updated[start:end]
        trigger_t = local_t[local_updated]
        if len(trigger_t) > 0:
            ax_bottom.scatter(
                trigger_t,
                np.full_like(trigger_t, y_level, dtype=float),
                marker='|',
                s=190,
                linewidths=1.2,
                color=color,
                alpha=0.95,
            )

    ax_bottom.set_yticks([1.0, 2.0, 3.0])
    ax_bottom.set_yticklabels(['TTC-RL-1', 'ST-ETC', 'ET-PRL'], fontsize=9)
    ax_bottom.set_ylim(0.5, 3.5)
    ax_bottom.set_xlim(float(local_t[0]), float(local_t[-1]) if len(local_t) > 1 else float(local_t[0] + 1.0))
    ax_bottom.set_xlabel('Time of day (h)', fontsize=10)
    ax_bottom.set_ylabel('Trigger pulses', fontsize=10)
    ax_bottom.grid(axis='x', color='#ECECEC', linewidth=0.6)
    ax_bottom.grid(axis='y', color='#F3F3F3', linewidth=0.5)
    ax_bottom.text(0.01, 0.98, '(b)', transform=ax_bottom.transAxes, va='top', ha='left', fontsize=10, fontweight='bold')
    _style_axes(ax_bottom)

    for hour in range(0, 25, 2):
        ax_bottom.axvline(hour, color='#F1F1F1', linewidth=0.5, zorder=0)

    fig.suptitle(
        f'Typical-day load-trigger alignment (Day {selected_day}, {sampling_interval_min:.0f}-min sampling)',
        fontsize=11,
        y=0.98,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format='svg', dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved figure: {output_path}')
    print(f'Selected day: {selected_day}')
    print(f'High-change threshold |ΔQ_load| (q={high_change_quantile:.2f}): {high_thr:.3f}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Figure 5-2-6 trigger-load alignment.')
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--env-data-path', type=Path, default=None)
    parser.add_argument('--output-path', type=Path, default=None)
    parser.add_argument('--sampling-interval-min', type=float, default=5.0)
    parser.add_argument('--selected-day', type=int, default=None)
    parser.add_argument('--high-change-quantile', type=float, default=0.85)
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    results_dir = args.results_dir or (root / 'logs' / 'control_compare' / '20260402_222140')
    env_data_path = args.env_data_path or (root / 'data' / 'dqn' / 'test_data.csv')
    output_path = args.output_path or (root / 'docs' / 'pics' / 'fig_5_2_6_trigger_load_alignment.svg')

    generate_trigger_alignment_figure(
        results_dir=results_dir,
        env_data_path=env_data_path,
        output_path=output_path,
        sampling_interval_min=args.sampling_interval_min,
        selected_day=args.selected_day,
        high_change_quantile=args.high_change_quantile,
    )


if __name__ == '__main__':
    main()
