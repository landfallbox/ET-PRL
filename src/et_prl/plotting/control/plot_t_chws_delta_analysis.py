"""Generate Figure 5-2-4: redesigned delta-action analysis.

Panel content:
- (a) Local disturbance + absolute T_chws step response in a typical window
    (highlight event-triggered ZOH holding behavior)
- (b) Full-period delta-frequency distribution in log scale
    (discrete x-axis from -3 to 3 with tail merge)

Usage:
    uv run python src/plotting/plot_t_chws_delta_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from et_prl.config.base import project_root
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch


plt.rcParams.update(
    {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'axes.unicode_minus': True,
    }
)


COLOR_BASE = '#242424'
COLOR_ST = '#1F4E79'
COLOR_ET = '#E64B35'
COLOR_DISTURB = '#2E8B9A'
COLOR_TRIGGER = '#808080'


def load_control_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load strategy CSVs for difference analysis."""
    from et_prl.plotting.control.results_path import resolve_results_dir

    results_dir = resolve_results_dir(results_dir)
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


def _prepare_action_series(
    results: dict[str, pd.DataFrame],
    sampling_interval_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    et_steps, et_values = _load_series(results, 'ET-PRL')
    ttc_steps, ttc_values = _load_series(results, 'TTC-RL-1')
    st_steps, st_values = _load_series(results, 'ST-ETC')

    common_length = min(len(et_steps), len(et_values), len(ttc_steps), len(ttc_values), len(st_steps), len(st_values))
    if common_length == 0:
        raise ValueError('Control result series are empty.')

    time_hours = et_steps[:common_length] * sampling_interval_min / 60.0
    et_actions = et_values[:common_length]
    ttc_actions = ttc_values[:common_length]
    st_actions = st_values[:common_length]
    delta_et = et_actions - ttc_actions
    delta_st = st_actions - ttc_actions
    return time_hours, et_actions, ttc_actions, st_actions, delta_et, delta_st


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis='both', labelsize=6.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def _darken_color(color: str, factor: float = 0.72) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    darker = np.clip(rgb * factor, 0.0, 1.0)
    return mcolors.to_hex(darker)


def _load_disturbance_series(env_data_path: Path, target_length: int) -> tuple[np.ndarray, str]:
    if not env_data_path.exists():
        print(f'Warning: disturbance data not found at {env_data_path}')
        return np.zeros(target_length, dtype=float), 'Disturbance (N/A)'

    env_df = pd.read_csv(env_data_path)
    if 'CL' in env_df.columns:
        values = env_df['CL'].to_numpy(dtype=float)
        return values[:target_length], 'Cooling load (kW)'
    if 'Twb' in env_df.columns:
        values = env_df['Twb'].to_numpy(dtype=float)
        return values[:target_length], r'Wet-bulb temperature ($^{\circ}$C)'

    first_col = env_df.columns[0]
    values = env_df[first_col].to_numpy(dtype=float)
    return values[:target_length], f'Disturbance ({first_col})'


def _extract_trigger_onset_times(et_df: pd.DataFrame, sampling_interval_min: float) -> np.ndarray:
    steps = et_df['step'].to_numpy(dtype=float)
    action_values = pd.to_numeric(et_df['action_value'], errors='coerce').to_numpy(dtype=float)

    # Action-change onset is the most trustworthy visual anchor for setpoint jumps.
    action_change_idx = np.where(np.abs(np.diff(action_values)) > 1e-9)[0] + 1

    logged_onset_idx: list[np.ndarray] = []
    for col in ('gate_signal', 'action_updated'):
        if col not in et_df.columns:
            continue
        raw = pd.to_numeric(et_df[col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        gate = raw > 0.5
        if not np.any(gate):
            continue
        onset = np.where(gate & np.concatenate(([True], ~gate[:-1])))[0]
        if onset.size > 0:
            logged_onset_idx.append(onset)

    if logged_onset_idx:
        logged = np.unique(np.concatenate(logged_onset_idx))
        if action_change_idx.size == 0:
            final_idx = logged
        else:
            unmatched = 0
            for idx in action_change_idx:
                if np.min(np.abs(logged - idx)) > 1:
                    unmatched += 1
            mismatch_ratio = unmatched / float(action_change_idx.size)

            # If logged trigger fields miss many action changes, merge action-change onsets.
            if mismatch_ratio > 0.20:
                print(
                    'Warning: trigger logs are not fully aligned with action changes '
                    f'(mismatch_ratio={mismatch_ratio:.2f}); using merged onsets for visualization.'
                )
                final_idx = np.unique(np.concatenate([logged, action_change_idx]))
            else:
                final_idx = logged
    else:
        final_idx = action_change_idx

    if final_idx.size == 0:
        return np.asarray([], dtype=float)
    return steps[final_idx] * sampling_interval_min / 60.0


def _count_action_changes(values: np.ndarray, atol: float = 1e-9) -> int:
    if len(values) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(values)) > atol))


def _compute_chattering_rate(values: np.ndarray, atol: float = 1e-9) -> float:
    """Immediate back-and-forth toggles like a,b,a indicate chattering."""
    if len(values) < 3:
        return 0.0
    prev_vals = values[:-2]
    curr_vals = values[1:-1]
    next_vals = values[2:]
    toggles = (~np.isclose(curr_vals, prev_vals, atol=atol, rtol=0.0)) & np.isclose(
        next_vals,
        prev_vals,
        atol=atol,
        rtol=0.0,
    )
    return float(np.mean(toggles))


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lower = float(np.min(arr))
    upper = float(np.max(arr))
    if upper - lower < 1e-12:
        return np.zeros_like(arr)
    return (arr - lower) / (upper - lower)


def _post_step_values_at(times: np.ndarray, values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Sample a post-step series value at arbitrary query times."""
    if len(times) == 0 or len(values) == 0 or len(query_times) == 0:
        return np.asarray([], dtype=float)
    idx = np.searchsorted(times, query_times, side='right') - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def _select_typical_window_zoh(
    time_hours: np.ndarray,
    et_actions: np.ndarray,
    ttc_actions: np.ndarray,
    disturbance_values: np.ndarray,
    window_hours: float = 20.0,
    step_hours: float = 1.0,
    min_reconstruction_ratio: float = 0.15,
) -> tuple[float, float]:
    """Select a representative window emphasizing ZOH stability and meaningful disturbance."""
    if len(time_hours) == 0:
        raise ValueError('No data available for local-window selection.')
    if window_hours <= 0.0 or step_hours <= 0.0:
        raise ValueError('window_hours and step_hours must be positive.')

    start_min = float(np.min(time_hours))
    start_max = float(np.max(time_hours) - window_hours)
    if start_max <= start_min:
        return float(np.min(time_hours)), float(np.max(time_hours))

    all_candidates: list[dict[str, float]] = []
    starts = np.arange(start_min, start_max + 1e-9, step_hours, dtype=float)
    for start in starts:
        end = start + window_hours
        mask = (time_hours >= start) & (time_hours <= end)
        count = int(np.count_nonzero(mask))
        if count < 6:
            continue

        local_et = et_actions[mask]
        local_ttc = ttc_actions[mask]
        local_dist = disturbance_values[mask]

        disturbance_range = float(np.max(local_dist) - np.min(local_dist))
        hold_length = float(count / (_count_action_changes(local_et) + 1))
        reconstruction_ratio = float(np.mean(~np.isclose(local_et, local_ttc, atol=1e-9, rtol=0.0)))
        chattering_rate = float(_compute_chattering_rate(local_et))

        all_candidates.append(
            {
                'start': float(start),
                'end': float(end),
                'disturbance_range': disturbance_range,
                'hold_length': hold_length,
                'reconstruction_ratio': reconstruction_ratio,
                'chattering_rate': chattering_rate,
            }
        )

    if not all_candidates:
        raise ValueError('No valid candidate windows found.')

    candidates = [
        item for item in all_candidates if item['reconstruction_ratio'] >= min_reconstruction_ratio
    ]
    if not candidates:
        print(
            'Warning: no windows satisfy minimum reconstruction ratio '
            f"{min_reconstruction_ratio:.2f}; falling back to all candidates."
        )
        candidates = all_candidates

    ranges = _minmax_normalize(np.asarray([item['disturbance_range'] for item in candidates], dtype=float))
    holds = _minmax_normalize(np.asarray([item['hold_length'] for item in candidates], dtype=float))
    recons = _minmax_normalize(np.asarray([item['reconstruction_ratio'] for item in candidates], dtype=float))
    chatters = _minmax_normalize(np.asarray([item['chattering_rate'] for item in candidates], dtype=float))

    scores = 0.55 * holds + 0.25 * ranges + 0.20 * recons - 0.35 * chatters
    best_idx = int(np.argmax(scores))
    best = candidates[best_idx]

    selected_start = float(np.floor(best['start']))
    selected_end = float(np.floor(best['end']))
    print(
        'Auto-selected panel-a window (ZOH-oriented): '
        f"{selected_start:.1f}h-{selected_end:.1f}h "
        f"(raw {best['start']:.1f}h-{best['end']:.1f}h) | "
        f"score={scores[best_idx]:.4f}, "
        f"hold={best['hold_length']:.2f}, "
        f"recon={best['reconstruction_ratio']:.3f}, "
        f"range={best['disturbance_range']:.2f}, "
        f"chatter={best['chattering_rate']:.4f}"
    )
    return selected_start, selected_end


def generate_delta_figure(
    results: dict[str, pd.DataFrame],
    output_path: Path,
    sampling_interval_min: float = 5.0,
    micro_x_min: Optional[float] = None,
    micro_x_max: Optional[float] = None,
    env_data_path: Optional[Path] = None,
) -> None:
    """Generate and save Figure 5-2-5 delta analysis."""
    required = {'ET-PRL', 'ST-ETC', 'TTC-RL-1'}
    if not required.issubset(results):
        missing = sorted(required - set(results.keys()))
        raise KeyError(f'Missing required strategies: {", ".join(missing)}')

    time_hours, et_actions, ttc_actions, _st_actions, delta_et, delta_st = _prepare_action_series(
        results,
        sampling_interval_min,
    )
    et_df = results['ET-PRL']

    if env_data_path is None:
        env_data_path = project_root() / 'data' / 'dqn' / 'test_data.csv'

    disturbance_values, disturbance_label = _load_disturbance_series(env_data_path, len(time_hours))
    common_length = min(len(time_hours), len(disturbance_values), len(et_df))
    time_hours = time_hours[:common_length]
    et_actions = et_actions[:common_length]
    ttc_actions = ttc_actions[:common_length]
    delta_et = delta_et[:common_length]
    delta_st = delta_st[:common_length]
    disturbance_values = disturbance_values[:common_length]
    et_df = et_df.iloc[:common_length]
    trigger_times = _extract_trigger_onset_times(et_df, sampling_interval_min)

    if micro_x_min is None and micro_x_max is None:
        micro_x_min, micro_x_max = _select_typical_window_zoh(
            time_hours,
            et_actions,
            ttc_actions,
            disturbance_values,
            window_hours=20.0,
            step_hours=1.0,
        )
    elif micro_x_min is None or micro_x_max is None:
        raise ValueError('micro_x_min and micro_x_max must be provided together.')

    fig = plt.figure(figsize=(7.06, 4.28))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.22)
    left = outer[0].subgridspec(2, 1, height_ratios=[0.84, 1.16], hspace=0.05)
    ax_a_top = fig.add_subplot(left[0])
    ax_a_bottom = fig.add_subplot(left[1], sharex=ax_a_top)
    ax_b = fig.add_subplot(outer[1])

    micro_mask = (time_hours >= micro_x_min) & (time_hours <= micro_x_max)
    if not np.any(micro_mask):
        raise ValueError('Selected local window does not overlap with available data.')

    local_t = time_hours[micro_mask]
    local_et_action = et_actions[micro_mask]
    local_ttc_action = ttc_actions[micro_mask]
    local_disturbance = disturbance_values[micro_mask]
    trigger_local = trigger_times[(trigger_times >= micro_x_min) & (trigger_times <= micro_x_max)]
    tick_start = int(np.floor(micro_x_min))
    tick_end = int(np.floor(micro_x_max))
    if tick_end <= tick_start:
        tick_end = tick_start + 1
    x_ticks = np.arange(tick_start, tick_end + 1, 5, dtype=float)
    if x_ticks.size < 2:
        x_ticks = np.linspace(tick_start, tick_end, 2, dtype=float)

    # Upper stack: physical disturbance context.
    ax_a_top.plot(
        local_t,
        local_disturbance,
        color=COLOR_DISTURB,
        linewidth=0.8,
        alpha=0.95,
        zorder=3,
    )

    ax_a_top.set_ylabel(disturbance_label, fontsize=6.5)
    _style_axes(ax_a_top)
    ax_a_top.grid(False)
    ax_a_top.grid(axis='y', color='#EBEBEB', linestyle='-', linewidth=0.55, alpha=1.0)
    ax_a_top.tick_params(axis='x', labelbottom=False)
    fig.text(0.095, 0.955, '(a)', fontsize=6.5, fontweight='bold', va='top', ha='left')
    # Lower stack: absolute setpoint trajectories for visualizing ZOH hold behavior.

    ax_a_bottom.step(
        local_t,
        local_ttc_action,
        where='post',
        color=COLOR_BASE,
        linewidth=1.0,
        linestyle='--',
        alpha=0.50,
        zorder=4.5,
        label='TTC-RL-1 (time-triggered baseline)',
    )

    ax_a_bottom.step(
        local_t,
        local_et_action,
        where='post',
        color=COLOR_ET,
        linewidth=0.6,
        alpha=0.95,
        zorder=5.0,
        label='ET-PRL (event-driven ZOH)',
    )

    y_low = float(min(np.min(local_ttc_action), np.min(local_et_action)) - 0.5)
    y_high = float(max(np.max(local_ttc_action), np.max(local_et_action)) + 0.5)
    tick_start = int(np.floor(y_low))
    tick_end = int(np.ceil(y_high))
    y_ticks = np.arange(tick_start, tick_end + 1, 1)
    ax_a_bottom.set_xlim(micro_x_min, micro_x_max)
    ax_a_bottom.set_ylim(y_low, y_high)
    ax_a_bottom.set_yticks(y_ticks)
    ax_a_bottom.set_xticks(x_ticks)
    ax_a_bottom.set_xlabel('Time (h)', fontsize=6.5)
    ax_a_bottom.set_ylabel(r'$T_{\mathrm{chws}}$ setpoint ($^{\circ}$C)', fontsize=6.5)
    _style_axes(ax_a_bottom)
    ax_a_bottom.grid(False)
    ax_a_bottom.grid(axis='y', color='#EBEBEB', linestyle='-', linewidth=0.55, alpha=1.0)

    # Draw trigger connectors strictly between the cooling-load curve and ET setpoint curve.
    if trigger_local.size > 0:
        trigger_top_values = np.interp(trigger_local, local_t, local_disturbance)
        trigger_bottom_values = _post_step_values_at(local_t, local_et_action, trigger_local)
        for t_mark, y_top, y_bottom in zip(trigger_local, trigger_top_values, trigger_bottom_values):
            connector = ConnectionPatch(
                xyA=(float(t_mark), float(y_top)),
                coordsA=ax_a_top.transData,
                xyB=(float(t_mark), float(y_bottom)),
                coordsB=ax_a_bottom.transData,
                axesA=ax_a_top,
                axesB=ax_a_bottom,
                color=COLOR_TRIGGER,
                linestyle='--',
                linewidth=0.4,
                alpha=0.7,
                zorder=3,
                clip_on=False,
            )
            fig.add_artist(connector)

    # Panel (a) legend above the left subplot (figure coordinates for alignment).
    fig.legend(
        handles=[
            Line2D([0], [0], color=COLOR_DISTURB, linewidth=0.8, label='Cooling load'),
            Line2D([0], [0], color=COLOR_TRIGGER, linewidth=0.4, linestyle='--', alpha=0.7,
                label='ET trigger'),
            Line2D([0], [0], color=COLOR_BASE, linewidth=1.0, linestyle='--', alpha=0.50,
                   label='TTC-RL-1'),
            Line2D([0], [0], color=COLOR_ET, linewidth=0.6, linestyle='-', label='ET-PRL'),
        ],
        loc='upper center',
        bbox_to_anchor=(0.33, 0.96),
        ncol=4,
        fontsize=5.5,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    plot_values = np.arange(-3, 4, 1, dtype=int)
    clipped_et = np.clip(delta_et, plot_values[0], plot_values[-1]).astype(int)
    clipped_st = np.clip(delta_st, plot_values[0], plot_values[-1]).astype(int)
    et_counts = np.array([(clipped_et == val).sum() for val in plot_values], dtype=float)
    st_counts = np.array([(clipped_st == val).sum() for val in plot_values], dtype=float)

    bar_width = 0.36
    x = plot_values.astype(float)
    ax_b.bar(
        x - bar_width / 2.0,
        st_counts,
        width=bar_width,
        color=COLOR_ST,
        alpha=0.68,
        edgecolor=_darken_color(COLOR_ST),
        linewidth=0.2,
        zorder=2,
        label='ST-ETC vs TTC-RL-1',
    )
    ax_b.bar(
        x + bar_width / 2.0,
        et_counts,
        width=bar_width,
        color=COLOR_ET,
        alpha=0.78,
        edgecolor=_darken_color(COLOR_ET),
        linewidth=0.2,
        zorder=3,
        label='ET-PRL vs TTC-RL-1',
    )

    ax_b.set_yscale('log')
    ax_b.set_xlim(plot_values[0] - 0.8, plot_values[-1] + 0.8)
    ax_b.set_xticks(plot_values)
    ax_b.set_xlabel(r'$\Delta T_{\mathrm{chws}}$ ($^{\circ}$C)', fontsize=6.5)
    ax_b.set_ylabel('Frequency (log scale)', fontsize=6.5)
    _style_axes(ax_b)
    ax_b.grid(False)
    ax_b.grid(axis='y', color='#EBEBEB', linestyle='-', linewidth=0.55, alpha=1.0)
    fig.text(0.67, 0.955, '(b)', fontsize=6.5, fontweight='bold', va='top', ha='left')

    # Panel (b) legend above the right subplot (figure coordinates for alignment).
    fig.legend(
        handles=[
            Patch(facecolor=COLOR_ST, edgecolor=_darken_color(COLOR_ST), linewidth=0.45, alpha=0.68,
                  label='ST-ETC'),
            Patch(facecolor=COLOR_ET, edgecolor=_darken_color(COLOR_ET), linewidth=0.45, alpha=0.78,
                  label='ET-PRL'),
        ],
        loc='upper center',
        bbox_to_anchor=(0.82, 0.96),
        ncol=2,
        fontsize=5.5,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='svg', dpi=100, pad_inches=0.02)
    plt.close(fig)
    print(f'Saved Figure 5-2-4 to {output_path}')


def main(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    env_data_path: Optional[Path] = None,
    micro_x_min: Optional[float] = None,
    micro_x_max: Optional[float] = None,
) -> None:
    if results_dir is None:
        from et_prl.plotting.control.results_path import default_control_compare_results_dir

        results_dir = default_control_compare_results_dir()
    if output_dir is None:
        output_dir = project_root() / 'outputs' / 'figures'
    if env_data_path is None:
        env_data_path = project_root() / 'data' / 'dqn' / 'test_data.csv'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading control results from {results_dir}...')
    results = load_control_results(results_dir)
    output_path = output_dir / 'fig7_t_chws_delta_analysis.svg'
    generate_delta_figure(
        results,
        output_path,
        env_data_path=env_data_path,
        micro_x_min=micro_x_min,
        micro_x_max=micro_x_max,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 5-2-4 T_chws delta analysis.')
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--env-data-path', type=Path, default=None)
    parser.add_argument('--micro-x-min', type=float, default=None)
    parser.add_argument('--micro-x-max', type=float, default=None)
    args = parser.parse_args()
    main(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        env_data_path=args.env_data_path,
        micro_x_min=args.micro_x_min,
        micro_x_max=args.micro_x_max,
    )
