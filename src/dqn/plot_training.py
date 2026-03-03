"""训练过程可视化模块 —— Fixed-step DQN 基准训练曲线（论文图）"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ─────────────────────────── 样式常量 ───────────────────────────
_COLORS = {
    "train_reward": "#2166ac",
    "val_reward":   "#d6604d",
    "comfort":      "#4dac26",
    "energy":       "#b8860b",
    "loss":         "#7b2d8b",
    "best":         "#d73027",
}
_ALPHA_FILL = 0.12
_FIG_DPI = 150
_FONT_SIZE_LABEL = 11
_FONT_SIZE_TICK  = 9
_FONT_SIZE_TITLE = 12
_FONT_SIZE_LEGEND = 9


def _latest_train_dir(log_root: Path | None = None) -> Path:
    """在 logs/dqn/train/ 下找最新的实验目录（按目录名时间戳排序）。"""
    root = (log_root or Path("logs")) / "dqn" / "train"
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"在 {root} 下找不到任何训练实验目录")
    return dirs[-1]


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """对一维数组做等宽移动平均平滑（边界使用有效均值）。"""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def plot_dqn_training_curves(
    train_experiment_dir: Path | None = None,
    output_dir: Path | None = None,
    smooth_window: int = 3,
    save: bool = True,
    show: bool = False,
) -> Path | None:
    """
    绘制 Fixed-step DQN 训练动态双子图并保存。

    子图 (a)：训练累积奖励 + 验证奖励随 Episode 变化。
    子图 (b)：训练集平均舒适度分数与平均能效分数随 Episode 变化。

    Parameters
    ----------
    train_experiment_dir : Path, optional
        训练实验目录（含 training_history.csv）；默认自动找最新目录。
    output_dir : Path, optional
        图片输出目录；默认与 train_experiment_dir 相同。
    smooth_window : int
        移动平均窗口长度（用于平滑训练奖励曲线）。
    save : bool
        是否保存图片。
    show : bool
        是否调用 plt.show() 显示图片。

    Returns
    -------
    Path | None
        保存的图片路径，若 save=False 则返回 None。
    """
    # ── 确定实验目录与数据路径 ──────────────────────────────────
    exp_dir = train_experiment_dir or _latest_train_dir()
    history_path = exp_dir / "training_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"训练历史文件不存在: {history_path}")

    df = pd.read_csv(history_path)
    episodes = df["epoch"].to_numpy()
    train_reward = df["train_reward"].to_numpy()
    train_comfort = df["train_avg_comfort_score"].to_numpy()
    train_energy  = df["train_avg_energy_score"].to_numpy()
    train_loss    = df["train_loss"].to_numpy()

    # 验证奖励仅在 checkpoint episode 有值
    val_mask   = df["val_reward"].notna()
    val_ep     = episodes[val_mask]
    val_reward = df["val_reward"].to_numpy()[val_mask]

    # 找最优 checkpoint episode
    best_idx      = int(np.argmax(val_reward))
    best_ep       = val_ep[best_idx]
    best_val_r    = val_reward[best_idx]

    # 以 "每步平均奖励" 规范化训练累积奖励，使量纲与验证奖励对齐
    train_steps = df["train_steps"].to_numpy()
    val_steps   = df["val_steps"].bfill().to_numpy()
    # 同时在原始量纲绘图（累积奖励差异量级过大，改用归一化）
    train_r_norm = train_reward / train_steps
    val_r_norm   = val_reward   / val_steps[val_mask]
    best_val_r_norm = best_val_r / val_steps[val_mask][best_idx]

    train_r_smooth = _smooth(train_r_norm, smooth_window)

    # ── 全局字体配置 ────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ── 创建画布 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 4.2),
        dpi=_FIG_DPI,
        gridspec_kw={"wspace": 0.38},
    )

    # ════════════════════════════════════════════════════════════
    # 子图 (a)：每步平均训练奖励 + 验证奖励
    # ════════════════════════════════════════════════════════════
    ax = axes[0]

    # 原始训练奖励（细线 + 低透明度）
    ax.plot(
        episodes, train_r_norm,
        color=_COLORS["train_reward"], alpha=0.25,
        linewidth=0.8, zorder=2,
    )
    # 平滑后的训练奖励（粗线）
    ax.plot(
        episodes, train_r_smooth,
        color=_COLORS["train_reward"], linewidth=1.8,
        label="Train reward (smoothed)", zorder=3,
    )

    # 验证奖励点
    ax.scatter(
        val_ep, val_r_norm,
        color=_COLORS["val_reward"], s=40, zorder=5,
        label="Val reward (checkpoint)", edgecolors="white", linewidths=0.5,
    )
    # 连接验证点的折线
    ax.plot(
        val_ep, val_r_norm,
        color=_COLORS["val_reward"], linewidth=1.0,
        alpha=0.55, zorder=4,
    )

    # 标注最优 checkpoint
    ax.axvline(best_ep, color=_COLORS["best"], linewidth=1.2,
               linestyle="--", alpha=0.7, zorder=3)
    ax.scatter(
        [best_ep], [best_val_r_norm],
        color=_COLORS["best"], s=100, zorder=6,
        marker="*", label=f"Best (Ep {best_ep}, val={best_val_r:.1f})",
    )

    # fill between smooth train curve 与 0 的区间（视觉强调）
    ax.fill_between(
        episodes, train_r_smooth, train_r_smooth.min() * 0.97,
        alpha=_ALPHA_FILL, color=_COLORS["train_reward"],
    )

    ax.set_xlabel("Episode", fontsize=_FONT_SIZE_LABEL)
    ax.set_ylabel("Avg Reward per Step", fontsize=_FONT_SIZE_LABEL)
    ax.set_title("(a) Training & Validation Reward", fontsize=_FONT_SIZE_TITLE, pad=8)
    ax.legend(fontsize=_FONT_SIZE_LEGEND, framealpha=0.85, loc="lower right")
    ax.tick_params(labelsize=_FONT_SIZE_TICK)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # ════════════════════════════════════════════════════════════
    # 子图 (b)：平均舒适度 & 能效分数
    # ════════════════════════════════════════════════════════════
    ax2 = axes[1]

    comfort_smooth = _smooth(train_comfort, smooth_window)
    energy_smooth  = _smooth(train_energy,  smooth_window)

    # 原始（浅色）
    ax2.plot(episodes, train_comfort, color=_COLORS["comfort"],
             alpha=0.20, linewidth=0.8, zorder=2)
    ax2.plot(episodes, train_energy,  color=_COLORS["energy"],
             alpha=0.20, linewidth=0.8, zorder=2)

    # 平滑（深色）
    ax2.plot(episodes, comfort_smooth,
             color=_COLORS["comfort"], linewidth=1.8, zorder=3,
             label="Comfort score (smoothed)")
    ax2.plot(episodes, energy_smooth,
             color=_COLORS["energy"],  linewidth=1.8, zorder=3,
             linestyle="--", label="Energy score (smoothed)")

    # 参考零线
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

    ax2.set_xlabel("Episode", fontsize=_FONT_SIZE_LABEL)
    ax2.set_ylabel("Avg Score per Step", fontsize=_FONT_SIZE_LABEL)
    ax2.set_title("(b) Comfort & Energy Score Dynamics", fontsize=_FONT_SIZE_TITLE, pad=8)
    ax2.legend(fontsize=_FONT_SIZE_LEGEND, framealpha=0.85, loc="lower right")
    ax2.tick_params(labelsize=_FONT_SIZE_TICK)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    # ── 图总标题 & 保存 ──────────────────────────────────────────
    fig.suptitle(
        "Fixed-step DQN Baseline — Training Dynamics",
        fontsize=_FONT_SIZE_TITLE + 1, y=1.01, fontweight="bold",
    )
    fig.tight_layout()

    save_path: Path | None = None
    if save:
        out_dir = output_dir or exp_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "dqn_training_curves.png"
        fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
        print(f"[plot_training] 图片已保存: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return save_path
