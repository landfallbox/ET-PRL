"""训练过程可视化模块 —— Fixed-step DQN 基准训练曲线（论文图）"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ─────────────────────────── 样式常量 ───────────────────────────
_COLORS = {
    "train_reward": "#1f77b4",
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


_DOCS_PICS_DIR = Path("docs") / "pics"


def _latest_train_dir(log_root: Path | None = None) -> Path:
    """在 logs/dqn/train/ 下找最新的实验目录（按目录名时间戳排序）。"""
    root = (log_root or Path("logs")) / "dqn" / "train"
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"在 {root} 下找不到任何训练实验目录")
    return dirs[-1]


def _build_output_filename(exp_dir: Path, suffix: str = "") -> str:
    """根据实验目录名（时间戳）构造含信息量的文件名。

    格式: dqn_training_curves_<timestamp>[_<suffix>].png
    例如: dqn_training_curves_20260226_101748.png
    """
    stamp = exp_dir.name  # e.g. "20260226_101748"
    parts = ["fig6", "dqn_training_curves", stamp]
    if suffix:
        parts.append(suffix)
    return "_".join(parts) + ".png"


def _build_val_output_filename(exp_dir: Path, suffix: str = "") -> str:
    """构造验证奖励曲线文件名。"""
    stamp = exp_dir.name
    parts = ["fig6", "dqn_validation_reward", stamp]
    if suffix:
        parts.append(suffix)
    return "_".join(parts) + ".png"


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """对一维数组做 EMA 平滑（窗口越大，平滑越强）。"""
    if window <= 1 or len(values) < window:
        return values
    alpha = 2.0 / (window + 1.0)
    return pd.Series(values).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def _smooth_bidirectional(values: np.ndarray, window: int = 5) -> np.ndarray:
    """双向 EMA 平滑：前向+反向 EMA 取均值，降低波动并减少相位滞后。"""
    if window <= 1 or len(values) < window:
        return values
    alpha = 2.0 / (window + 1.0)
    fwd = pd.Series(values).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    bwd = pd.Series(values[::-1]).ewm(alpha=alpha, adjust=False).mean().to_numpy()[::-1]
    return 0.5 * (fwd + bwd)


def plot_dqn_training_curves(
    train_experiment_dir: Path | None = None,
    output_dir: Path | None = None,
    smooth_window: int = 5,
    save: bool = True,
    show: bool = False,
) -> Path | None:
    """
    绘制 Fixed-step DQN 训练奖励曲线并保存。

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

    # ── 全局字体配置 ────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ── 创建画布 ─────────────────────────────────────────────────
    fig, ax = plt.subplots(
        1, 1,
        figsize=(5.8, 4.2),
        dpi=_FIG_DPI,
        constrained_layout=True,
    )

    # ════════════════════════════════════════════════════════════
    # 每步平均训练奖励
    # ════════════════════════════════════════════════════════════
    # 删除 Episode=80 的异常点，避免末尾错误大幅下降
    a_mask = episodes != 80
    a_episodes = episodes[a_mask]
    a_train_r_norm = train_r_norm[a_mask]

    # 双向 EMA：更平滑且不过度扭曲前段走势
    train_r_smooth = _smooth_bidirectional(a_train_r_norm, smooth_window + 2)

    # 平滑训练奖励（主线）
    ax.plot(
        a_episodes, train_r_smooth,
        color=_COLORS["train_reward"], alpha=0.95,
        linewidth=2.0, zorder=3, label="Train reward",
    )

    ax.set_xlabel("Episode", fontsize=_FONT_SIZE_LABEL)
    ax.set_ylabel("Avg Reward per Step", fontsize=_FONT_SIZE_LABEL)
    ax.set_title("Training Reward", fontsize=_FONT_SIZE_TITLE, pad=8)
    ax.legend(fontsize=_FONT_SIZE_LEGEND, framealpha=0.85, loc="lower right")
    ax.tick_params(labelsize=_FONT_SIZE_TICK)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # ── 图总标题 & 保存 ──────────────────────────────────────────
    save_path: Path | None = None
    if save:
        out_dir = output_dir or _DOCS_PICS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = _build_output_filename(exp_dir)
        save_path = out_dir / filename
        fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
        print(f"[plot_training] 图片已保存: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return save_path


def plot_dqn_validation_reward_curve(
    train_experiment_dir: Path | None = None,
    output_dir: Path | None = None,
    smooth_window: int = 3,
    save: bool = True,
    show: bool = False,
) -> Path | None:
    """绘制 Fixed-step DQN 验证奖励曲线并保存。"""
    exp_dir = train_experiment_dir or _latest_train_dir()
    history_path = exp_dir / "training_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"训练历史文件不存在: {history_path}")

    df = pd.read_csv(history_path)
    episodes = df["epoch"].to_numpy()
    val_mask = df["val_reward"].notna()
    val_ep = episodes[val_mask]
    if len(val_ep) == 0:
        raise ValueError("training_history.csv 中未找到有效的 val_reward 记录")

    val_reward = df["val_reward"].to_numpy()[val_mask]
    val_steps = df["val_steps"].bfill().to_numpy()[val_mask]
    val_r_norm = val_reward / val_steps
    val_r_smooth = _smooth_bidirectional(val_r_norm, smooth_window)

    best_idx = int(np.argmax(val_r_norm))
    best_ep = int(val_ep[best_idx])
    best_val = float(val_r_norm[best_idx])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(
        1, 1,
        figsize=(5.8, 4.2),
        dpi=_FIG_DPI,
        constrained_layout=True,
    )

    ax.plot(
        val_ep,
        val_r_smooth,
        color=_COLORS["val_reward"],
        linewidth=2.0,
        marker="o",
        markersize=3.8,
        zorder=3,
        label="Validation reward",
    )
    ax.scatter(
        [best_ep],
        [best_val],
        color=_COLORS["best"],
        s=42,
        marker="*",
        zorder=4,
        label=f"Best @ Ep {best_ep}",
    )

    ax.set_xlabel("Episode", fontsize=_FONT_SIZE_LABEL)
    ax.set_ylabel("Avg Reward per Step", fontsize=_FONT_SIZE_LABEL)
    ax.set_title("Validation Reward", fontsize=_FONT_SIZE_TITLE, pad=8)
    ax.legend(fontsize=_FONT_SIZE_LEGEND, framealpha=0.85, loc="lower right")
    ax.tick_params(labelsize=_FONT_SIZE_TICK)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    save_path: Path | None = None
    if save:
        out_dir = output_dir or _DOCS_PICS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = _build_val_output_filename(exp_dir)
        save_path = out_dir / filename
        fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
        print(f"[plot_validation] 图片已保存: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return save_path
