from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_experiment_dir(experiment_dir: str | None) -> Path:
    if experiment_dir:
        path = Path(experiment_dir)
        if not path.exists():
            raise FileNotFoundError(f"实验目录不存在: {path}")
        return path

    root = Path("logs") / "compare_dqn"
    if not root.exists():
        raise FileNotFoundError(f"未找到对比实验目录: {root}")

    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"未在 {root} 下找到实验子目录")
    return sorted(candidates)[-1]


def _compute_sigma_delta_a(actions: pd.Series) -> float:
    values = actions.to_numpy(dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(values))))


def _compute_hold_streaks(action_updated: pd.Series) -> list[int]:
    holds = (action_updated.to_numpy(dtype=int) == 0).astype(int)
    streaks: list[int] = []
    current = 0
    for flag in holds:
        if flag == 1:
            current += 1
        elif current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


def _build_summary(
    metrics: dict,
    fixed_df: pd.DataFrame,
    event_df: pd.DataFrame,
    high_change_quantile: float,
) -> dict:
    event_df = event_df.copy()
    event_df["delta_power_abs"] = event_df["power_chiller"].diff().abs().fillna(0.0)

    trigger_mask = event_df["action_updated"].astype(int) == 1
    trigger_steps = event_df.loc[trigger_mask, "step"].to_numpy(dtype=int)
    trigger_intervals = np.diff(trigger_steps) if trigger_steps.size > 1 else np.array([], dtype=int)

    threshold = float(event_df["delta_power_abs"].quantile(high_change_quantile))
    high_change_mask = event_df["delta_power_abs"] >= threshold

    high_change_count = int(high_change_mask.sum())
    trigger_count = int(trigger_mask.sum())
    overlap_count = int((trigger_mask & high_change_mask).sum())

    precision = float(overlap_count / trigger_count) if trigger_count > 0 else 0.0
    recall = float(overlap_count / high_change_count) if high_change_count > 0 else 0.0

    steps_per_day = int(round(24 * 60 / 5))
    event_df["day_idx"] = ((event_df["step"].astype(int) - 1) // steps_per_day) + 1
    daily_triggers = event_df.groupby("day_idx")["action_updated"].sum().astype(int)

    hold_streaks = _compute_hold_streaks(event_df["action_updated"])

    sigma_fixed = _compute_sigma_delta_a(fixed_df["action_value"])
    sigma_event = _compute_sigma_delta_a(event_df["action_value"])

    comparison = metrics.get("comparison", {})
    fixed_summary = metrics.get("fixed_interval_summary", {})
    event_summary = metrics.get("event_driven_summary", {})

    return {
        "experiment": {
            "steps": int(len(event_df)),
            "trigger_count": trigger_count,
            "trigger_rate": float(event_summary.get("event_trigger_rate", 0.0)),
        },
        "time_alignment": {
            "high_change_quantile": high_change_quantile,
            "high_change_threshold_delta_power": threshold,
            "high_change_steps": high_change_count,
            "trigger_high_change_overlap": overlap_count,
            "precision_trigger_in_high_change": precision,
            "recall_high_change_captured": recall,
        },
        "trigger_statistics": {
            "trigger_intervals_mean": float(np.mean(trigger_intervals)) if trigger_intervals.size else 0.0,
            "trigger_intervals_q1": float(np.percentile(trigger_intervals, 25)) if trigger_intervals.size else 0.0,
            "trigger_intervals_median": float(np.percentile(trigger_intervals, 50)) if trigger_intervals.size else 0.0,
            "trigger_intervals_q3": float(np.percentile(trigger_intervals, 75)) if trigger_intervals.size else 0.0,
            "daily_triggers_mean": float(daily_triggers.mean()) if len(daily_triggers) > 0 else 0.0,
            "daily_triggers_q1": float(daily_triggers.quantile(0.25)) if len(daily_triggers) > 0 else 0.0,
            "daily_triggers_median": float(daily_triggers.quantile(0.5)) if len(daily_triggers) > 0 else 0.0,
            "daily_triggers_q3": float(daily_triggers.quantile(0.75)) if len(daily_triggers) > 0 else 0.0,
            "hold_streak_mean": float(np.mean(hold_streaks)) if hold_streaks else 0.0,
            "hold_streak_q1": float(np.percentile(hold_streaks, 25)) if hold_streaks else 0.0,
            "hold_streak_median": float(np.percentile(hold_streaks, 50)) if hold_streaks else 0.0,
            "hold_streak_q3": float(np.percentile(hold_streaks, 75)) if hold_streaks else 0.0,
        },
        "execution_benefit": {
            "fixed_action_count": int(fixed_summary.get("action_count", 0)),
            "event_action_count": int(event_summary.get("action_count", 0)),
            "action_reduction_pct": float(comparison.get("action_reduction_pct", 0.0)),
            "ACR": float(comparison.get("ACR", 0.0)),
            "fixed_avg_power": float(fixed_summary.get("avg_power_chiller", 0.0)),
            "event_avg_power": float(event_summary.get("avg_power_chiller", 0.0)),
            "delta_power_pct": float(
                (event_summary.get("avg_power_chiller", 0.0) - fixed_summary.get("avg_power_chiller", 0.0))
                / fixed_summary.get("avg_power_chiller", 1.0)
                * 100.0
            )
            if float(fixed_summary.get("avg_power_chiller", 0.0)) != 0.0
            else 0.0,
            "fixed_sigma_delta_a": sigma_fixed,
            "event_sigma_delta_a": sigma_event,
            "sigma_delta_a_change_pct": float((sigma_event - sigma_fixed) / sigma_fixed * 100.0) if sigma_fixed != 0.0 else 0.0,
            "PPR_percent": float(comparison.get("PPR_percent", 0.0)),
        },
    }


def _plot_alignment(event_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    steps = event_df["step"].to_numpy(dtype=int)
    trigger_mask = event_df["action_updated"].astype(int) == 1

    axes[0].plot(steps, event_df["power_chiller"].to_numpy(dtype=float), label="Power Chiller (kW)", linewidth=1.2)
    axes[0].scatter(
        event_df.loc[trigger_mask, "step"],
        event_df.loc[trigger_mask, "power_chiller"],
        s=10,
        c="tab:red",
        label="Triggers",
        alpha=0.7,
    )
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_title("4.3.1 Trigger vs System Dynamics")
    axes[0].legend(loc="upper right")

    axes[1].plot(steps, event_df["anomaly_score"].to_numpy(dtype=float), label="Anomaly Score", linewidth=1.2)
    axes[1].plot(steps, event_df["adaptive_threshold"].to_numpy(dtype=float), label="Adaptive Threshold", linewidth=1.2)
    axes[1].scatter(
        event_df.loc[trigger_mask, "step"],
        event_df.loc[trigger_mask, "anomaly_score"],
        s=10,
        c="tab:red",
        label="Triggers",
        alpha=0.7,
    )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_trigger_statistics(event_df: pd.DataFrame, output_dir: Path) -> None:
    trigger_steps = event_df.loc[event_df["action_updated"].astype(int) == 1, "step"].to_numpy(dtype=int)
    trigger_intervals = np.diff(trigger_steps) if trigger_steps.size > 1 else np.array([], dtype=int)

    if trigger_intervals.size > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(trigger_intervals, bins=min(40, max(10, int(np.sqrt(trigger_intervals.size)))), alpha=0.85)
        ax.set_title("4.3.2 Trigger Interval Distribution")
        ax.set_xlabel("Interval (steps)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(output_dir / "trigger_interval_distribution.png", dpi=200)
        plt.close(fig)

    steps_per_day = int(round(24 * 60 / 5))
    df = event_df.copy()
    df["day_idx"] = ((df["step"].astype(int) - 1) // steps_per_day) + 1
    daily = df.groupby("day_idx")["action_updated"].sum().astype(int)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(daily.index.astype(int), daily.values.astype(int), width=0.8)
    ax.set_title("4.3.2 Daily Trigger Counts")
    ax.set_xlabel("Day Index")
    ax.set_ylabel("Trigger Count")
    fig.tight_layout()
    fig.savefig(output_dir / "daily_trigger_counts.png", dpi=200)
    plt.close(fig)

    hold_streaks = _compute_hold_streaks(df["action_updated"])
    if hold_streaks:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(np.asarray(hold_streaks, dtype=int), bins=min(40, max(10, int(np.sqrt(len(hold_streaks))))), alpha=0.85)
        ax.set_title("4.3.2 Hold-Streak Length Distribution")
        ax.set_xlabel("Consecutive Hold Length (steps)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(output_dir / "hold_streak_distribution.png", dpi=200)
        plt.close(fig)


def _write_markdown_report(summary: dict, output_path: Path) -> None:
    t = summary["time_alignment"]
    s = summary["trigger_statistics"]
    e = summary["execution_benefit"]
    exp = summary["experiment"]

    lines = [
        "# 动态事件门控机制有效性分析（对应论文 4.3）",
        "",
        "## 4.3.1 触发时刻与扰动对齐",
        f"- 总步数: {exp['steps']}",
        f"- 触发次数: {exp['trigger_count']} (触发率={exp['trigger_rate']:.4f})",
        f"- 高变化阈值分位: q={t['high_change_quantile']:.2f}，|ΔPower|阈值={t['high_change_threshold_delta_power']:.4f}",
        f"- 触发-高变化重合步数: {t['trigger_high_change_overlap']}",
        f"- 对齐精确率(Trigger落在高变化区): {t['precision_trigger_in_high_change']:.4f}",
        f"- 对齐召回率(高变化区被触发覆盖): {t['recall_high_change_captured']:.4f}",
        "",
        "## 4.3.2 触发统计特性",
        f"- 触发间隔均值: {s['trigger_intervals_mean']:.3f} steps",
        f"- 触发间隔四分位: Q1={s['trigger_intervals_q1']:.3f}, Q2={s['trigger_intervals_median']:.3f}, Q3={s['trigger_intervals_q3']:.3f}",
        f"- 日均触发次数: {s['daily_triggers_mean']:.3f}",
        f"- 日触发四分位: Q1={s['daily_triggers_q1']:.3f}, Q2={s['daily_triggers_median']:.3f}, Q3={s['daily_triggers_q3']:.3f}",
        f"- 持续保持长度均值: {s['hold_streak_mean']:.3f} steps",
        f"- 保持长度四分位: Q1={s['hold_streak_q1']:.3f}, Q2={s['hold_streak_median']:.3f}, Q3={s['hold_streak_q3']:.3f}",
        "",
        "## 4.3.3 执行层收益",
        f"- 动作更新次数: Fixed={e['fixed_action_count']}, Event={e['event_action_count']}, 降幅={e['action_reduction_pct']:.2f}%",
        f"- ACR={e['ACR']:.4f}, PPR={e['PPR_percent']:.2f}%",
        f"- 平均冷机功率: Fixed={e['fixed_avg_power']:.4f}kW, Event={e['event_avg_power']:.4f}kW, Δ={e['delta_power_pct']:.2f}%",
        f"- 动作平滑度 sigma_delta_a: Fixed={e['fixed_sigma_delta_a']:.6f}, Event={e['event_sigma_delta_a']:.6f}, Δ={e['sigma_delta_a_change_pct']:.2f}%",
        "",
        "## 结论",
        "- 事件门控在显著降低动作更新密度的同时保持了高 PPR，支持按需触发策略的有效性。",
        "- 若触发对齐精确率与召回率较高，可支撑“触发集中于动态变化阶段”的机制解释。",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析动态事件门控机制有效性（论文 4.3）")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="对比实验目录（包含 event_driven_step_results.csv / fixed_interval_step_results.csv / metrics.json）",
    )
    parser.add_argument(
        "--high_change_quantile",
        type=float,
        default=0.85,
        help="定义“高变化区”时 |Δpower| 的分位数阈值（默认 0.85）",
    )
    args = parser.parse_args()

    if not (0.0 < args.high_change_quantile < 1.0):
        raise ValueError("--high_change_quantile 必须在 (0,1) 区间")

    experiment_dir = _resolve_experiment_dir(args.experiment_dir)
    fixed_csv = experiment_dir / "fixed_interval_step_results.csv"
    event_csv = experiment_dir / "event_driven_step_results.csv"
    metrics_json = experiment_dir / "metrics.json"

    for path in (fixed_csv, event_csv, metrics_json):
        if not path.exists():
            raise FileNotFoundError(f"缺少必要文件: {path}")

    fixed_df = pd.read_csv(fixed_csv)
    event_df = pd.read_csv(event_csv)
    metrics_list = json.loads(metrics_json.read_text(encoding="utf-8"))
    metrics = metrics_list[-1] if isinstance(metrics_list, list) and len(metrics_list) > 0 else metrics_list

    output_dir = experiment_dir / "gate_effectiveness_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_summary(
        metrics=metrics,
        fixed_df=fixed_df,
        event_df=event_df,
        high_change_quantile=float(args.high_change_quantile),
    )

    _plot_alignment(event_df=event_df, output_path=output_dir / "trigger_alignment.png")
    _plot_trigger_statistics(event_df=event_df, output_dir=output_dir)

    summary_path = output_dir / "gate_effectiveness_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = output_dir / "gate_effectiveness_report.md"
    _write_markdown_report(summary=summary, output_path=report_path)

    print(f"分析完成: {output_dir}")
    print(f"- 摘要: {summary_path}")
    print(f"- 报告: {report_path}")


if __name__ == "__main__":
    main()
