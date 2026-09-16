"""生成触发时刻与负荷扰动对齐分析图（论文 4.3.1）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from et_prl.config.base import project_root


def _resolve_experiment_dir(experiment_dir: str | None) -> Path:
    if experiment_dir:
        path = Path(experiment_dir)
        if not path.exists():
            raise FileNotFoundError(f"实验目录不存在: {path}")
        return path

    root = project_root() / "outputs" / "runs" / "control_compare"
    if not root.exists():
        raise FileNotFoundError(f"未找到对比实验目录: {root}")

    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"未在 {root} 下找到实验子目录")
    return sorted(candidates)[-1]


def _validate_columns(df: pd.DataFrame) -> None:
    required_cols = {
        "step",
        "action_updated",
        "power_chiller",
        "anomaly_score",
        "adaptive_threshold",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"event_driven_step_results.csv 缺少必要列: {sorted(missing)}")


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy(dtype=float)


def _merge_short_runs(values: np.ndarray, min_run_length: int) -> np.ndarray:
    if min_run_length <= 1 or values.size == 0:
        return values

    merged = values.copy()
    idx = 0
    length = merged.size
    while idx < length:
        run_end = idx + 1
        while run_end < length and merged[run_end] == merged[idx]:
            run_end += 1
        run_length = run_end - idx
        if run_length < min_run_length:
            if idx > 0:
                fill_value = merged[idx - 1]
            elif run_end < length:
                fill_value = merged[run_end]
            else:
                fill_value = merged[idx]
            merged[idx:run_end] = fill_value
        idx = run_end
    return merged


def _merge_event_features(event_df: pd.DataFrame, event_features_csv: str | None) -> pd.DataFrame:
    if not event_features_csv:
        return event_df

    features_path = Path(event_features_csv)
    if not features_path.exists():
        raise FileNotFoundError(f"事件特征文件不存在: {features_path}")

    features_df = pd.read_csv(features_path)
    if "step" not in features_df.columns:
        raise ValueError("事件特征文件必须包含 step 列")

    return event_df.merge(features_df, on="step", how="left", suffixes=("", "_event"))


def _to_binary_state(series: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce").fillna(0.0)
    unique_vals = set(np.unique(cleaned.to_numpy(dtype=float)).tolist())
    if unique_vals.issubset({0.0, 1.0}):
        return cleaned.astype(int)
    threshold = float(cleaned.median())
    return (cleaned >= threshold).astype(int)


def _build_paper_event_mask(
    event_df: pd.DataFrame,
    lambda_col: str,
    occupancy_col: str,
    comfort_col: str,
    th_a: float,
    th_b: float,
    th_c: float,
    th_d: float,
) -> tuple[pd.Series, str]:
    notes: list[str] = []

    if comfort_col not in event_df.columns:
        raise ValueError(f"paper_events 模式缺少舒适度列: {comfort_col}")
    comfort = pd.to_numeric(event_df[comfort_col], errors="coerce").fillna(0.0)

    if lambda_col in event_df.columns:
        lambda_state = _to_binary_state(event_df[lambda_col])
    elif "energy_score" in event_df.columns:
        lambda_state = (pd.to_numeric(event_df["energy_score"], errors="coerce").fillna(0.0) <= 0.5).astype(int)
        notes.append("lambda 缺失，使用 energy_score<=0.5 作为高电价代理")
    else:
        lambda_state = pd.Series(np.zeros(len(event_df), dtype=int), index=event_df.index)
        notes.append("lambda 缺失，使用全低电价(0)代理")

    if occupancy_col in event_df.columns:
        occupancy_state = _to_binary_state(event_df[occupancy_col])
    elif "CL" in event_df.columns:
        occupancy_state = _to_binary_state(event_df["CL"])
        notes.append("occupancy 缺失，使用 CL 中位数二值化作为有人代理")
    elif "CL_predict" in event_df.columns:
        occupancy_state = _to_binary_state(event_df["CL_predict"])
        notes.append("occupancy 缺失，使用 CL_predict 中位数二值化作为有人代理")
    else:
        occupancy_state = pd.Series(np.ones(len(event_df), dtype=int), index=event_df.index)
        notes.append("occupancy 缺失，使用全有人(1)代理")

    event1 = lambda_state.ne(lambda_state.shift(1)).fillna(False)
    event2 = occupancy_state.ne(occupancy_state.shift(1)).fillna(False)
    event3 = (occupancy_state == 1) & (lambda_state == 0) & (comfort < th_a)
    event4 = (occupancy_state == 1) & (lambda_state == 1) & (comfort < th_b)
    event5 = (occupancy_state == 0) & (lambda_state == 0) & (comfort < th_c)
    event6 = (occupancy_state == 0) & (lambda_state == 1) & (comfort < th_d)

    mask = event1 | event2 | event3 | event4 | event5 | event6
    note_text = "；".join(notes) if notes else "使用显式 lambda/K/comfort 列计算"
    return mask.astype(bool), note_text


def _build_alignment_figure(
    df: pd.DataFrame,
    high_change_quantile: float,
    output_prefix: Path,
    dpi: int,
    smooth_window: int,
    green_rule: str,
    lambda_col: str,
    occupancy_col: str,
    comfort_col: str,
    th_a: float,
    th_b: float,
    th_c: float,
    th_d: float,
    logic_band_columns: int,
    logic_min_run_bins: int,
    logic_both_min_ratio: float,
    logic_green_min_ratio: float,
    logic_event_min_ratio: float,
) -> None:
    event_df = df.copy()
    event_df["delta_power_abs"] = event_df["power_chiller"].diff().abs().fillna(0.0)

    threshold = float(event_df["delta_power_abs"].quantile(high_change_quantile))
    if green_rule == "paper_events":
        high_change_mask, paper_note = _build_paper_event_mask(
            event_df=event_df,
            lambda_col=lambda_col,
            occupancy_col=occupancy_col,
            comfort_col=comfort_col,
            th_a=th_a,
            th_b=th_b,
            th_c=th_c,
            th_d=th_d,
        )
        region_label = "Paper-defined event region"
    else:
        high_change_mask = event_df["delta_power_abs"] >= threshold
        paper_note = ""
        region_label = "High-change region"
    trigger_mask = event_df["action_updated"].astype(int) == 1
    criterion_mask = event_df["anomaly_score"] > event_df["adaptive_threshold"]
    overlap_mask = trigger_mask & high_change_mask

    steps = event_df["step"].to_numpy(dtype=int)
    anomaly = event_df["anomaly_score"].to_numpy(dtype=float)
    adaptive_threshold = event_df["adaptive_threshold"].to_numpy(dtype=float)

    anomaly_smooth = _rolling_mean(anomaly, smooth_window)
    threshold_smooth = _rolling_mean(adaptive_threshold, smooth_window)

    criterion_true = int(criterion_mask.sum())
    trigger_true = int(trigger_mask.sum())
    criterion_consistency = float((criterion_mask == trigger_mask).mean())
    precision = float(overlap_mask.sum() / trigger_true) if trigger_true > 0 else 0.0
    recall = float(overlap_mask.sum() / int(high_change_mask.sum())) if int(high_change_mask.sum()) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 3.4), sharex=True)

    ax1.plot(steps, anomaly_smooth, linewidth=1.4, alpha=0.95, label=r"$A(x_t)$")
    ax1.plot(steps, threshold_smooth, linewidth=1.6, alpha=0.95, label=r"$\tau_t^*$")
    ax1.set_ylabel("Score")
    ax1.grid(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        borderaxespad=0.2,
    )

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 3.4), sharex=True)
    both_mask = overlap_mask
    trigger_only_mask = trigger_mask & (~high_change_mask)
    high_change_only_mask = high_change_mask & (~trigger_mask)
    category = np.zeros(len(event_df), dtype=int)
    category[trigger_only_mask.to_numpy(dtype=bool)] = 1
    category[high_change_only_mask.to_numpy(dtype=bool)] = 2
    category[both_mask.to_numpy(dtype=bool)] = 3

    target_band_columns = max(20, int(logic_band_columns))
    bin_size = max(1, int(np.ceil(len(category) / target_band_columns)))
    n_bins = int(np.ceil(len(category) / bin_size))
    category_binned = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, len(category))
        chunk = category[start:end]
        counts = np.bincount(chunk, minlength=4)
        chunk_size = max(1, int(chunk.size))
        event_ratio = float((counts[1] + counts[2] + counts[3]) / chunk_size)
        both_ratio = float(counts[3] / chunk_size)
        green_ratio = float(counts[2] / chunk_size)
        if event_ratio < float(logic_event_min_ratio):
            category_binned[i] = 0
        elif both_ratio >= float(logic_both_min_ratio):
            category_binned[i] = 3
        elif green_ratio >= float(logic_green_min_ratio):
            category_binned[i] = 2
        elif counts[1] > 0:
            category_binned[i] = 1
        elif counts[2] > 0:
            category_binned[i] = 2
        else:
            category_binned[i] = 0
    category_binned = _merge_short_runs(category_binned, max(1, int(logic_min_run_bins)))

    x_min = float(steps.min()) - 0.5
    x_max = float(steps.max()) + 0.5

    color_map = ListedColormap(["#f2f2f2", "tab:orange", "tab:green", "tab:purple"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], color_map.N)
    ax2.imshow(
        category_binned[np.newaxis, :],
        cmap=color_map,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        extent=[x_min, x_max, 0.0, 1.0],
        origin="lower",
    )

    ax2.set_ylim(-0.1, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_ylabel("Binary")
    ax2.legend(
        handles=[
            Patch(facecolor="tab:orange", label="Actual trigger"),
            Patch(facecolor="tab:green", label=region_label),
            Patch(facecolor="tab:purple", label="Both (trigger ∩ region)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        borderaxespad=0.2,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    panel1_path = output_prefix.parent / f"{output_prefix.name}_criterion.png"
    panel2_path = output_prefix.parent / f"{output_prefix.name}_logic.png"

    fig1.tight_layout(rect=(0, 0, 1, 0.96))
    fig2.tight_layout(rect=(0, 0, 1, 0.96))

    fig1.savefig(panel1_path, dpi=dpi)
    fig2.savefig(panel2_path, dpi=dpi)
    plt.close(fig1)
    plt.close(fig2)

    overlap_count = int((trigger_mask & high_change_mask).sum())
    print(f"图像已保存: {panel1_path}")
    print(f"图像已保存: {panel2_path}")
    if green_rule == "paper_events":
        print(f"绿色区域定义: 论文事件规则 (a,b,c,d)=({th_a:.3f},{th_b:.3f},{th_c:.3f},{th_d:.3f})")
        print(f"规则说明: {paper_note}")
    else:
        print(f"高变化阈值(|ΔP|): {threshold:.4f} (q={high_change_quantile:.2f})")
    print(f"绿色区域步数: {int(high_change_mask.sum())}")
    print(f"触发步数: {int(trigger_mask.sum())}")
    print(f"触发与绿色区域重合步数: {overlap_count}")
    print("对齐指标（用于正文描述）:")
    print(f"- Criterion hit ratio: {criterion_true / len(event_df):.3f}")
    print(f"- Trigger ratio: {trigger_true / len(event_df):.3f}")
    print(f"- Criterion-trigger consistency: {criterion_consistency:.3f}")
    print(f"- Precision: {precision:.3f}")
    print(f"- Recall: {recall:.3f}")
    print(f"- F1: {f1:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成触发时刻与负荷扰动对齐分析图")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="对比实验目录（含 event_driven_step_results.csv）；若不指定则自动选最新目录",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="outputs/figures/trigger_alignment",
        help="输出图片名前缀（默认 outputs/figures/trigger_alignment，将生成 *_criterion.png / *_logic.png）",
    )
    parser.add_argument(
        "--high_change_quantile",
        type=float,
        default=0.85,
        help="定义高变化区间的 |Δpower| 分位数阈值（默认 0.85）",
    )
    parser.add_argument(
        "--green_rule",
        type=str,
        default="delta_power",
        choices=["delta_power", "paper_events"],
        help="绿色区域定义方式：delta_power(原逻辑) 或 paper_events(论文事件规则)",
    )
    parser.add_argument(
        "--event_features_csv",
        type=str,
        default=None,
        help="额外事件特征文件（可选，需含 step 列；用于提供 lambda/K 等）",
    )
    parser.add_argument("--lambda_col", type=str, default="lambda_signal", help="电价状态列名（0低价/1高价）")
    parser.add_argument("--occupancy_col", type=str, default="occupancy_state", help="占用状态列名（0无人/1有人）")
    parser.add_argument("--comfort_col", type=str, default="comfort_score", help="舒适度列名")
    parser.add_argument("--th_a", type=float, default=0.4, help="论文阈值 a")
    parser.add_argument("--th_b", type=float, default=0.3, help="论文阈值 b")
    parser.add_argument("--th_c", type=float, default=0.2, help="论文阈值 c")
    parser.add_argument("--th_d", type=float, default=0.1, help="论文阈值 d")
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出图像分辨率（默认 300）",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=9,
        help="上图平滑窗口（默认 9）",
    )
    parser.add_argument(
        "--logic_band_columns",
        type=int,
        default=240,
        help="逻辑图横向分箱列数（默认 240，越大越零星）",
    )
    parser.add_argument(
        "--logic_min_run_bins",
        type=int,
        default=1,
        help="逻辑图最短色块长度（按分箱数，默认 1；1 表示不合并色块）",
    )
    parser.add_argument(
        "--logic_both_min_ratio",
        type=float,
        default=0.06,
        help="逻辑图分箱显示紫色的最小占比阈值（默认 0.06）",
    )
    parser.add_argument(
        "--logic_green_min_ratio",
        type=float,
        default=0.08,
        help="逻辑图分箱显示绿色的最小占比阈值（默认 0.08）",
    )
    parser.add_argument(
        "--logic_event_min_ratio",
        type=float,
        default=0.40,
        help="逻辑图分箱着色的最小事件占比阈值（默认 0.40，低于该值保留白色）",
    )
    args = parser.parse_args()

    if not (0.0 < args.high_change_quantile < 1.0):
        raise ValueError("--high_change_quantile 必须在 (0,1) 区间")
    if not (0.0 <= args.logic_both_min_ratio <= 1.0):
        raise ValueError("--logic_both_min_ratio 必须在 [0,1] 区间")
    if not (0.0 <= args.logic_green_min_ratio <= 1.0):
        raise ValueError("--logic_green_min_ratio 必须在 [0,1] 区间")
    if not (0.0 <= args.logic_event_min_ratio <= 1.0):
        raise ValueError("--logic_event_min_ratio 必须在 [0,1] 区间")

    experiment_dir = _resolve_experiment_dir(args.experiment_dir)
    from et_prl.plotting.control.results_path import resolve_results_dir

    event_csv = resolve_results_dir(experiment_dir) / "event_driven_step_results.csv"
    if not event_csv.exists():
        raise FileNotFoundError(f"缺少必要文件: {event_csv}")

    event_df = pd.read_csv(event_csv)
    event_df = _merge_event_features(event_df=event_df, event_features_csv=args.event_features_csv)
    _validate_columns(event_df)

    _build_alignment_figure(
        df=event_df,
        high_change_quantile=float(args.high_change_quantile),
        output_prefix=Path(args.output_prefix),
        dpi=int(args.dpi),
        smooth_window=max(1, int(args.smooth_window)),
        green_rule=str(args.green_rule),
        lambda_col=str(args.lambda_col),
        occupancy_col=str(args.occupancy_col),
        comfort_col=str(args.comfort_col),
        th_a=float(args.th_a),
        th_b=float(args.th_b),
        th_c=float(args.th_c),
        th_d=float(args.th_d),
        logic_band_columns=int(args.logic_band_columns),
        logic_min_run_bins=int(args.logic_min_run_bins),
        logic_both_min_ratio=float(args.logic_both_min_ratio),
        logic_green_min_ratio=float(args.logic_green_min_ratio),
        logic_event_min_ratio=float(args.logic_event_min_ratio),
    )


if __name__ == "__main__":
    main()
