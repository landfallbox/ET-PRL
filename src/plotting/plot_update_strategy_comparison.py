"""生成固定间隔/固定阈值/动态阈值三种更新策略对比图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _build_signal(time_axis: np.ndarray) -> np.ndarray:
    base = 0.42 * np.sin(1.35 * time_axis - 0.7)
    detail = 0.16 * np.sin(3.4 * time_axis + 0.2)
    trend = 0.05 * (time_axis - 5.0)
    return base + detail + trend


def _find_cross_trigger_points(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    abs_values = np.abs(values)
    trigger_mask = np.zeros_like(abs_values, dtype=bool)
    trigger_mask[0] = abs_values[0] >= threshold[0]
    trigger_mask[1:] = (abs_values[1:] >= threshold[1:]) & (abs_values[:-1] < threshold[:-1])
    return np.where(trigger_mask)[0]


def _find_intersections(time_axis: np.ndarray, signal: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find approximate intersection points (t, value) where signal crosses threshold.

    Uses linear interpolation between consecutive samples for a more precise location.
    Returns arrays (times, values).
    """
    diff = signal - threshold
    # detect intervals where sign changes or exact zero at left endpoint
    idx = np.where((diff[:-1] == 0) | (diff[:-1] * diff[1:] < 0))[0]
    times = []
    vals = []
    for i in idx:
        s0, s1 = signal[i], signal[i + 1]
        t0, t1 = time_axis[i], time_axis[i + 1]
        th0, th1 = threshold[i], threshold[i + 1]
        denom = (s1 - s0) - (th1 - th0)
        if denom == 0:
            u = 0.5
        else:
            u = (th0 - s0) / denom
        u = float(np.clip(u, 0.0, 1.0))
        t_cross = t0 + u * (t1 - t0)
        val_cross = s0 + u * (s1 - s0)
        times.append(t_cross)
        vals.append(val_cross)
    if len(times) == 0:
        return np.array([]), np.array([])
    return np.array(times), np.array(vals)


def _setup_chinese_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


def _draw_axes_with_arrows(ax: plt.Axes, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    ax.annotate(
        "",
        xy=(x_max, y_min + 0.02),
        xytext=(x_min, y_min + 0.02),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#2F2F2F"},
    )
    ax.annotate(
        "",
        xy=(x_min + 0.02, y_max),
        xytext=(x_min + 0.02, y_min),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#2F2F2F"},
    )


def _draw_bottom_timeline(ax: plt.Axes, trigger_t: np.ndarray, color: str, y_line: float) -> None:
    ax.hlines(y=y_line, xmin=0.8, xmax=9.2, colors="#9099A1", linewidth=1.3)
    
    # Handle overlapping points with vertical jitter
    # Group points by x-coordinate (within 0.1 tolerance)
    if len(trigger_t) == 0:
        return
    
    sorted_indices = np.argsort(trigger_t)
    sorted_t = trigger_t[sorted_indices]
    
    # Find groups of nearby points
    groups = []
    current_group = [sorted_t[0]]
    for i in range(1, len(sorted_t)):
        if abs(sorted_t[i] - sorted_t[i-1]) < 0.1:
            current_group.append(sorted_t[i])
        else:
            groups.append(current_group)
            current_group = [sorted_t[i]]
    groups.append(current_group)
    
    # Plot with vertical jitter for overlapping points
    jitter_step = 0.04
    for group in groups:
        group_arr = np.array(group)
        if len(group) == 1:
            # Single point - no jitter
            y_pos = np.full_like(group_arr, y_line)
        else:
            # Multiple overlapping points - apply vertical jitter
            n = len(group)
            offsets = np.linspace(-(n-1) * jitter_step / 2, (n-1) * jitter_step / 2, n)
            y_pos = y_line + offsets
        
        ax.scatter(group_arr, y_pos, s=220, color=color, edgecolors="white", linewidths=1.3, zorder=6)
        for x, y in zip(group_arr, y_pos):
            ax.text(x, y - 0.005, "v", ha="center", va="center", fontsize=9, color="white", zorder=7)


def _decorate_common_text(ax: plt.Axes, x_max: float, y_max: float, y_min: float) -> None:
    # Decorative labels removed as requested (no '高','低','0','偏差','时间 t','值')
    return


def create_update_strategy_comparison(output_path: Path, dpi: int = 220) -> Path:
    _setup_chinese_font()

    time_axis = np.linspace(0.8, 9.2, 500)
    signal = _build_signal(time_axis)

    y_min, y_max = -1.05, 1.05
    frame_colors = ["#8FB6DC", "#9ED5A7", "#B8ACE8"]
    signal_color = "#2E75C6"
    red = "#DF6C65"
    purple = "#8268C8"

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8), constrained_layout=True)

    interval_trigger_t = np.array([1.2, 2.5, 3.8, 5.1, 6.4, 7.7, 9.0])
    interval_trigger_y = np.interp(interval_trigger_t, time_axis, signal)

    delta_fixed = np.full_like(time_axis, 0.27)
    fixed_trigger_idx = _find_cross_trigger_points(signal, delta_fixed)
    fixed_trigger_t = time_axis[fixed_trigger_idx]
    fixed_trigger_y = signal[fixed_trigger_idx]

    delta_dynamic = 0.22 + 0.08 * np.sin(0.75 * time_axis + 0.5) + 0.04 * np.sin(2.1 * time_axis)
    dynamic_trigger_idx = _find_cross_trigger_points(signal, delta_dynamic)
    dynamic_trigger_t = time_axis[dynamic_trigger_idx]
    dynamic_trigger_y = signal[dynamic_trigger_idx]

    # compute interpolated intersections between curve and thresholds
    fixed_plus_t, fixed_plus_y = _find_intersections(time_axis, signal, delta_fixed)
    fixed_minus_t, fixed_minus_y = _find_intersections(time_axis, signal, -delta_fixed)

    dynamic_plus_t, dynamic_plus_y = _find_intersections(time_axis, signal, delta_dynamic)
    dynamic_minus_t, dynamic_minus_y = _find_intersections(time_axis, signal, -delta_dynamic)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(-1.45, 1.18)
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_axes_with_arrows(ax, x_min=0.6, x_max=9.4, y_min=y_min, y_max=y_max)

    signal_line = axes[0].plot(time_axis, signal, color=signal_color, linewidth=2.5)[0]
    for x in interval_trigger_t:
        axes[0].vlines(x, ymin=-1.24, ymax=np.interp(x, time_axis, signal), colors="#6E95CB", linestyles="--", linewidth=1.6, alpha=0.45)
    axes[0].scatter(interval_trigger_t, interval_trigger_y, s=84, color=signal_color, edgecolors=signal_color, zorder=5)
    _draw_bottom_timeline(axes[0], interval_trigger_t, color="#4A90E2", y_line=-1.24)
    _decorate_common_text(axes[0], x_max=9.4, y_max=y_max, y_min=y_min)
    interval_line = axes[0].plot([], [], linestyle="--", color="#6E95CB")[0]
    # legend marker for trigger points: use same style as bottom timeline in subplot 0
    trigger_marker = axes[0].scatter([], [], s=220, color="#4A90E2", edgecolors="white", linewidths=1.3)

    axes[1].plot(time_axis, signal, color=signal_color, linewidth=2.5)
    fixed_line = axes[1].plot(time_axis, delta_fixed, linestyle="--", color=red, linewidth=1.9)[0]
    axes[1].plot(time_axis, -delta_fixed, linestyle="--", color=red, linewidth=1.9)
    # draw precise intersection vlines and place bottom timeline markers (use subplot-0 style)
    fixed_all_t = np.concatenate([fixed_plus_t, fixed_minus_t])
    fixed_all_y = np.concatenate([fixed_plus_y, fixed_minus_y])
    for x, yv in zip(fixed_all_t, fixed_all_y):
        axes[1].vlines(x, ymin=-1.24, ymax=yv, colors="#6E95CB", linestyles="--", linewidth=1.6, alpha=0.45)
    axes[1].scatter(fixed_all_t, fixed_all_y, s=84, color="#4A90E2", edgecolors="#4A90E2", zorder=5)
    if fixed_all_t.size:
        _draw_bottom_timeline(axes[1], fixed_all_t, color="#4A90E2", y_line=-1.24)
    _decorate_common_text(axes[1], x_max=9.4, y_max=y_max, y_min=y_min)

    axes[2].plot(time_axis, signal, color=signal_color, linewidth=2.5)
    dynamic_line = axes[2].plot(time_axis, delta_dynamic, linestyle="--", color=purple, linewidth=1.9)[0]
    axes[2].plot(time_axis, -delta_dynamic, linestyle="--", color=purple, linewidth=1.9)
    # draw precise intersection vlines and place bottom timeline markers (use subplot-0 style)
    dynamic_all_t = np.concatenate([dynamic_plus_t, dynamic_minus_t])
    dynamic_all_y = np.concatenate([dynamic_plus_y, dynamic_minus_y])
    for x, yv in zip(dynamic_all_t, dynamic_all_y):
        axes[2].vlines(x, ymin=-1.24, ymax=yv, colors="#6E95CB", linestyles="--", linewidth=1.6, alpha=0.45)
    axes[2].scatter(dynamic_all_t, dynamic_all_y, s=84, color="#4A90E2", edgecolors="#4A90E2", zorder=5)
    if dynamic_all_t.size:
        _draw_bottom_timeline(axes[2], dynamic_all_t, color="#4A90E2", y_line=-1.24)
    _decorate_common_text(axes[2], x_max=9.4, y_max=y_max, y_min=y_min)

    # unified legend for signal, time-step marks, fixed and dynamic thresholds
    fig.legend(
        handles=[signal_line, fixed_line, dynamic_line, trigger_marker],
        labels=["室温偏差", "固定阈值", "动态阈值", "触发点"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=False,
        fontsize=11,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成三种更新策略对比示意图")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / "pics" / "fig1_update_strategy_comparison.svg",
        help="输出图片路径",
    )
    parser.add_argument("--dpi", type=int, default=220, help="导出图片 DPI")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = create_update_strategy_comparison(output_path=args.output, dpi=args.dpi)
    print(f"图像已生成: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
