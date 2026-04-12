"""Generate Figure 5-2-5: focused delta series and histogram analysis.

Panel content:
- (b) Focused delta time series in 100h-150h with 30-min mode resampling
- (c) Full-period delta histogram in log scale

Usage:
    uv run python src/plotting/generate_t_chws_fig_5_2_5_delta.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


plt.rcParams.update(
    {
        'font.family': 'Times New Roman',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
    }
)


COLOR_TTC = '#303030'
COLOR_ST = '#8A8A8A'


def load_control_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load strategy CSVs for difference analysis."""
    strategies = {
        'ET-PRL': 'event_driven_step_results.csv',
        'TTC-RL-1': 'fixed_interval_1_step_results.csv',
        'TTC-RL-2': 'fixed_interval_2_step_results.csv',
        'ST-ETC': 'event_triggered_etc_step_results.csv',
    }

    results: dict[str, pd.DataFrame] = {}
    for name, filename in strategies.items():
        file_path = results_dir / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            results[name] = df
            print(f'Loaded {name}: {len(df)} steps')
        else:
            print(f'Warning: {filename} not found at {file_path}')
    return results


def _load_series(results: dict[str, pd.DataFrame], strategy_name: str) -> tuple[np.ndarray, np.ndarray]:
    df = results[strategy_name]
    steps = df['step'].to_numpy(dtype=float)
    values = df['action_value'].to_numpy(dtype=float)
    return steps, values


def _select_representative_ttc(results: dict[str, pd.DataFrame], et_values: np.ndarray) -> str:
    candidates: dict[str, float] = {}
    for strategy_name in ('TTC-RL-1', 'TTC-RL-2'):
        if strategy_name not in results:
            continue
        _, candidate_values = _load_series(results, strategy_name)
        common_length = min(len(et_values), len(candidate_values))
        if common_length == 0:
            continue
        candidates[strategy_name] = float(np.mean(np.abs(et_values[:common_length] - candidate_values[:common_length])))

    if not candidates:
        raise KeyError('No TTC baseline is available for delta analysis.')
    return min(candidates, key=candidates.get)


def _prepare_difference_panel(
    results: dict[str, pd.DataFrame],
    sampling_interval_min: float,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray, str]], float, float]:
    et_steps, et_values = _load_series(results, 'ET-PRL')

    ttc_key = 'TTC-RL-1'
    if ttc_key not in results:
        ttc_key = _select_representative_ttc(results, et_values)
    ttc_steps, ttc_values = _load_series(results, ttc_key)

    if 'ST-ETC' not in results:
        raise KeyError('ST-ETC result is required for delta analysis.')
    st_steps, st_values = _load_series(results, 'ST-ETC')

    common_length = min(len(et_steps), len(et_values), len(ttc_steps), len(ttc_values), len(st_steps), len(st_values))
    if common_length == 0:
        raise ValueError('Control result series are empty.')

    time_hours = et_steps[:common_length] * sampling_interval_min / 60.0
    delta_ttc = et_values[:common_length] - ttc_values[:common_length]
    delta_st = et_values[:common_length] - st_values[:common_length]

    combined = np.concatenate([delta_ttc, delta_st])
    lower = float(np.percentile(combined, 1))
    upper = float(np.percentile(combined, 99))
    limit = max(abs(lower), abs(upper), 1.0)
    limit = float(np.ceil(limit * 2.0) / 2.0)

    series = [
        (f'ET-PRL - {ttc_key}', delta_ttc, COLOR_TTC),
        ('ET-PRL - ST-ETC', delta_st, COLOR_ST),
    ]
    return time_hours, series, -limit, limit


def _resample_mode_by_time_bin(
    time_hours: np.ndarray,
    values: np.ndarray,
    bin_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a discrete-valued series by per-bin mode with latest-value tie-break."""
    length = min(len(time_hours), len(values))
    if length == 0:
        return time_hours[:0], values[:0]

    t = time_hours[:length]
    v = values[:length]
    start_t = float(np.min(t))
    end_t = float(np.max(t))

    edges = np.arange(start_t, end_t + bin_hours, bin_hours, dtype=float)
    if edges.size < 2:
        edges = np.asarray([start_t, start_t + max(bin_hours, 1e-6)], dtype=float)
    elif edges[-1] < end_t:
        edges = np.append(edges, edges[-1] + bin_hours)

    bin_idx = np.digitize(t, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    centers = 0.5 * (edges[:-1] + edges[1:])

    out_t: list[float] = []
    out_v: list[float] = []
    for idx in range(len(edges) - 1):
        mask = bin_idx == idx
        if not np.any(mask):
            continue

        local_values = v[mask]
        uniq_vals, counts = np.unique(local_values, return_counts=True)
        max_count = int(np.max(counts))
        mode_candidates = uniq_vals[counts == max_count]

        if len(mode_candidates) == 1:
            chosen = float(mode_candidates[0])
        else:
            chosen = float(mode_candidates[0])
            for sample in local_values[::-1]:
                if np.any(np.isclose(sample, mode_candidates, rtol=0.0, atol=1e-9)):
                    chosen = float(sample)
                    break

        out_t.append(float(centers[idx]))
        out_v.append(chosen)

    return np.asarray(out_t, dtype=float), np.asarray(out_v, dtype=float)


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis='both', labelsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def generate_delta_figure(
    results: dict[str, pd.DataFrame],
    output_path: Path,
    sampling_interval_min: float = 5.0,
    micro_x_min: float = 100.0,
    micro_x_max: float = 150.0,
    micro_bin_hours: float = 0.5,
) -> None:
    """Generate and save Figure 5-2-5 delta analysis."""
    required = {'ET-PRL', 'ST-ETC'}
    if not required.issubset(results):
        missing = sorted(required - set(results.keys()))
        raise KeyError(f'Missing required strategies: {", ".join(missing)}')
    if 'TTC-RL-1' not in results and 'TTC-RL-2' not in results:
        raise KeyError('At least one TTC baseline is required (TTC-RL-1 or TTC-RL-2).')

    time_hours, diff_series, _, _ = _prepare_difference_panel(results, sampling_interval_min)
    y_min, y_max = -10.5, 10.5
    bins = np.linspace(y_min, y_max, 33)

    fig = plt.figure(figsize=(18.0, 7.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.5, 0.85], wspace=0.15)
    ax_b = fig.add_subplot(grid[0])
    ax_c = fig.add_subplot(grid[1], sharey=ax_b)

    for idx, (label, values, color) in enumerate(diff_series):
        mask = (time_hours >= micro_x_min) & (time_hours <= micro_x_max)
        local_t = time_hours[mask]
        local_v = values[mask]
        local_t_res, local_v_res = _resample_mode_by_time_bin(local_t, local_v, micro_bin_hours)

        if idx == 0:
            ax_b.step(
                local_t_res,
                local_v_res,
                where='mid',
                color=color,
                linewidth=2.0,
                alpha=0.95,
                linestyle='--',
                dashes=(3.0, 1.5),
                zorder=2,
            )
        else:
            ax_b.step(
                local_t_res,
                local_v_res,
                where='mid',
                color=color,
                linewidth=1.6,
                alpha=0.95,
                linestyle='-',
                zorder=3,
            )

    ax_b.set_xlim(micro_x_min, micro_x_max)
    ax_b.set_ylim(y_min, y_max)
    ax_b.set_yticks(np.arange(-10.0, 10.1, 5.0))
    ax_b.set_xticks(np.arange(micro_x_min, micro_x_max + 0.1, 10.0))
    ax_b.set_xlabel('Time (h)', fontsize=16, fontweight='bold')
    ax_b.set_ylabel(r'$\Delta T_{\mathrm{chws}}$ ($^{\circ}$C)', fontsize=16, fontweight='bold')
    _style_axes(ax_b)
    ax_b.grid(False)
    ax_b.grid(axis='y', color='#B8B8B8', linestyle='--', linewidth=0.6, alpha=0.55)
    ax_b.text(0.0, 1.02, '(b)', transform=ax_b.transAxes,
              fontsize=16, fontweight='bold', va='bottom', ha='left', clip_on=False)
    ax_b.text(0.99, 1.02, 'Focused window: 100h-150h (30-min mode)', transform=ax_b.transAxes,
              fontsize=11, va='bottom', ha='right', clip_on=False)

    hist_labels = [item[0] for item in diff_series]
    hist_data = [item[1] for item in diff_series]
    hist_colors = [item[2] for item in diff_series]
    counts = [np.histogram(values, bins=bins)[0] for values in hist_data]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_height = bins[1] - bins[0]
    bar_height = 0.40 * bin_height
    offset = 0.23 * bin_height

    ax_c.barh(
        bin_centers - offset,
        counts[0],
        height=bar_height,
        color=hist_colors[0],
        alpha=0.75,
        edgecolor='black',
        linewidth=0.4,
        hatch='//',
        zorder=2,
    )
    ax_c.barh(
        bin_centers + offset,
        counts[1],
        height=bar_height,
        color=hist_colors[1],
        alpha=0.60,
        edgecolor='black',
        linewidth=0.4,
        hatch='..',
        zorder=3,
    )

    max_count = max(int(np.max(values)) for values in counts if len(values) > 0)
    x_max = max(max_count * 1.45, 2.0)
    ax_c.set_xscale('log')
    ax_c.set_xlim(0.8, x_max)
    ax_c.set_xlabel('Frequency (log10 scale)', fontsize=16, fontweight='bold')
    log_ticks = [tick for tick in [1, 10, 100, 1000, 10000] if tick <= x_max]
    if log_ticks:
        ax_c.set_xticks(log_ticks)
        ax_c.set_xticklabels([str(int(tick)) for tick in log_ticks], fontsize=13)
    ax_c.tick_params(axis='x', labelsize=13)
    ax_c.set_yticks(ax_b.get_yticks())
    ax_c.yaxis.tick_right()
    ax_c.tick_params(axis='y', labelsize=12, labelleft=False, left=False, labelright=True, right=True)
    ax_c.grid(False)
    ax_c.grid(axis='y', color='#B8B8B8', linestyle='--', linewidth=0.6, alpha=0.55)
    ax_c.spines['top'].set_visible(True)
    ax_c.spines['right'].set_visible(True)
    ax_c.spines['left'].set_linewidth(1.1)
    ax_c.spines['bottom'].set_linewidth(1.1)
    ax_c.spines['top'].set_linewidth(1.1)
    ax_c.spines['right'].set_linewidth(1.1)
    ax_c.text(0.0, 1.02, '(c)', transform=ax_c.transAxes,
              fontsize=16, fontweight='bold', va='bottom', ha='left', clip_on=False)

    legend_handles = [
        Patch(facecolor=COLOR_TTC, edgecolor='black', hatch='//', alpha=0.75, label='ET-PRL - TTC-RL-1'),
        Patch(facecolor=COLOR_ST, edgecolor='black', hatch='..', alpha=0.60, label='ET-PRL - ST-ETC'),
    ]
    ax_c.legend(
        handles=legend_handles,
        loc='upper right',
        ncol=1,
        fontsize=10.5,
        framealpha=0.94,
        handlelength=1.8,
        borderaxespad=0.2,
    )

    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='svg', dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved Figure 5-2-5 to {output_path}')


def main(results_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> None:
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / 'logs' / 'control_compare' / '20260402_222140'
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'docs' / 'pics'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading control results from {results_dir}...')
    results = load_control_results(results_dir)
    output_path = output_dir / 'fig_5_2_5_t_chws_delta_analysis.svg'
    generate_delta_figure(results, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 5-2-5 T_chws delta analysis.')
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    main(results_dir=args.results_dir, output_dir=args.output_dir)
