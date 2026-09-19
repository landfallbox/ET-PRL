"""Generate Figure 5-2-7: ablation Pareto scatter.

从 outputs/runs/ablation/<ablation_id>/<timestamp>/metrics.json 读取各消融实验
的最新一次运行结果，绘制 Pareto 散点图。

Usage:
    uv run python src/et_prl/plotting/ablation/plot_ablation_pareto.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from et_prl.config.base import project_root
from et_prl.plotting._style import apply_paper_style

apply_paper_style(unicode_minus=True)


PROPOSED_METHOD = "Full ET-PRL"

# ablation_id -> 图中方法名
ABLATION_METHOD_NAMES: dict[str, str] = {
    "wo_dual_threshold": "Local Threshold Only",
    "wo_local_threshold": "Global Threshold Only",
    "single_scale_short": "Short-scale Only",
    "single_scale_medium": "Medium-scale Only",
    "single_scale_long": "Long-scale Only",
    "full_et_prl": PROPOSED_METHOD,
}


def _latest_metrics_path(ablation_root: Path, ablation_id: str) -> Path:
    """返回指定消融实验最新一次运行的 metrics.json 路径。"""
    run_dir = ablation_root / ablation_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"消融实验目录不存在: {run_dir}")

    run_timestamps = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    if not run_timestamps:
        raise FileNotFoundError(f"消融实验无运行记录: {run_dir}")

    metrics_path = run_dir / run_timestamps[-1] / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"消融实验缺少 metrics.json: {metrics_path}")
    return metrics_path


def _build_ablation_dataframe(ablation_root: Path) -> pd.DataFrame:
    """从各消融实验的最新 metrics.json 提取 (E_daily, PPR, ACR)。"""
    records = []
    for ablation_id, method in ABLATION_METHOD_NAMES.items():
        metrics_path = _latest_metrics_path(ablation_root, ablation_id)
        with open(metrics_path, encoding="utf-8") as f:
            entries = json.load(f)
        if not entries:
            raise ValueError(f"metrics.json 为空: {metrics_path}")
        latest = entries[-1]

        ed_summary = latest["event_driven_summary"]
        comparison = latest["comparison"]
        records.append(
            {
                "method": method,
                "E_daily": float(ed_summary["E_daily_kwh_per_day"]),
                "PPR": float(comparison["PPR_percent"]),
                "ACR": float(comparison["ACR"]),
            }
        )
    return pd.DataFrame.from_records(records)


def generate_figure(output_path: Path, ablation_root: Path) -> None:
    df = _build_ablation_dataframe(ablation_root)

    acr_min = float(df["ACR"].min())
    acr_max = float(df["ACR"].max())

    fig, ax = plt.subplots(figsize=(7.06, 4.28))
    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.13, top=0.94)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#D9D9D9", linestyle="--", linewidth=0.70, alpha=0.27)

    norm = mcolors.Normalize(vmin=acr_min, vmax=acr_max)
    cmap = plt.get_cmap("YlGnBu")
    scatter = ax.scatter(
        df["E_daily"],
        df["PPR"],
        c=df["ACR"],
        s=112.0,
        cmap=cmap,
        norm=norm,
        alpha=1.00,
        edgecolors="#263238",
        linewidths=0.55,
        zorder=3,
    )

    connector_df = df.sort_values(by=["E_daily", "PPR"], ascending=[True, False], kind="mergesort")
    ax.plot(
        connector_df["E_daily"],
        connector_df["PPR"],
        color="#3E4C59",
        linewidth=1.55,
        linestyle=(0, (4, 2)),
        alpha=0.95,
        zorder=2,
    )

    offsets: dict[str, tuple[float, float]] = {
        "Local Threshold Only": (-10.0, 10.0),
        "Global Threshold Only": (10.0, -6.0),
        "Short-scale Only": (8.0, -4.0),
        "Medium-scale Only": (-12.0, 8.0),
        "Long-scale Only": (-12.0, 6.0),
        PROPOSED_METHOD: (8.0, -8.0),
    }

    for _, row in df.iterrows():
        method = str(row["method"])
        x = float(row["E_daily"])
        y = float(row["PPR"])
        dx, dy = offsets[method]
        ax.annotate(
            method,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            fontweight="semibold" if method == PROPOSED_METHOD else "normal",
            color="#1F1F1F",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.98,
            },
            zorder=6,
        )

    ax.set_xlabel(r"Energy consumption $E_{\mathrm{daily}}$ (kWh/day)", fontsize=9)
    ax.set_ylabel("PPR (%)", fontsize=9, labelpad=6)
    ax.tick_params(axis="both", labelsize=8.5)

    x_margin = 24.0
    y_margin = 0.60
    x_min = float(df["E_daily"].min()) - x_margin
    x_max = max(7520.0, float(df["E_daily"].max()) + x_margin)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(float(df["PPR"].min()) - y_margin, float(df["PPR"].max()) + y_margin)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.055, pad=0.01)
    cbar.set_label("ACR", fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=7.5)

    pareto_handle = Line2D(
        [0],
        [0],
        color="#4E5B6A",
        linewidth=1.45,
        linestyle=(0, (4, 2)),
        label="All-point connector",
    )
    legend_main = ax.legend(
        handles=[pareto_handle],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.05),
        frameon=True,
        fontsize=7.5,
        title=None,
        borderpad=0.55,
        handlelength=1.9,
    )
    legend_main.get_frame().set_facecolor("white")
    legend_main.get_frame().set_edgecolor("#D4D9DE")
    legend_main.get_frame().set_alpha(1.00)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=140, pad_inches=0.02)
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 5-2-7 ablation Pareto scatter.")
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    root = project_root()
    output_path = args.output_path or (
        root / "outputs" / "figures" / "fig10_ablation_pareto_scatter.svg"
    )
    ablation_root = root / "outputs" / "runs" / "ablation"
    generate_figure(output_path=output_path, ablation_root=ablation_root)


if __name__ == "__main__":
    main()
