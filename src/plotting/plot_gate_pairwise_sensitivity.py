"""Generate publication-quality pairwise gate parameter sensitivity figures."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
    }
)

ROOT = Path("logs/online_anomaly_detection/sensitivity")
OUTPUT = Path("docs/pics/fig12_gate_key_parameter_sensitivity_response_curves.svg")

PARAM_LABELS_ALPHA = {
    "threshold_bias": r"$b_{\mathrm{bias}}$",
    "trigger_hysteresis_margin": r"$m_{\mathrm{hys}}$",
    "local_window_size": r"$W$",
    "score_short_weight": r"$w_{\mathrm{s}}$",
    "threshold_quantile": r"$q$",
}

METRICS = [
    ("R_test", "total_reward", r"$R_{\mathrm{test}}$"),
    ("N_daily", "N_daily_count_per_day", r"$N_{\mathrm{daily}}$"),
    ("E_daily", "E_daily_kwh_per_day", r"$E_{\mathrm{daily}}$"),
]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "blue_white_red",
    [
        (0.0, "#1d4ed8"),
        (0.25, "#60a5fa"),
        (0.5, "#ffffff"),
        (0.75, "#f87171"),
        (1.0, "#b91c1c"),
    ],
)


def _latest_csv(group: str) -> tuple[Path, pd.DataFrame]:
    candidates = sorted(ROOT.rglob(f"gate_pairwise_{group}_results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"找不到 {group} 结果 CSV，请先运行 pairwise 敏感性实验")
    data = pd.read_csv(candidates[0])
    return candidates[0], data


def _heatmap_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric_col: str,
    panel_label: str,
    metric_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    xs = sorted(data[x_col].unique())
    ys = sorted(data[y_col].unique())
    grid = np.full((len(ys), len(xs)), np.nan)
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    for _, row in data.iterrows():
        xi = x_idx[row[x_col]]
        yi = y_idx[row[y_col]]
        grid[yi, xi] = row[metric_col]

    if vmin is None:
        vmin = float(np.nanmin(grid))
    if vmax is None:
        vmax = float(np.nanmax(grid))

    center = np.nanmedian(grid)
    if vmin < center < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(
        grid,
        cmap=HEATMAP_CMAP,
        norm=norm,
        aspect="auto",
        origin="lower",
    )

    x_labels = [f"{v:g}" for v in xs]
    y_labels = [f"{v:g}" for v in ys]
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels(y_labels, fontsize=7)

    for i in range(len(ys)):
        for j in range(len(xs)):
            val = grid[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, f"{val:.1f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if abs(val - center) < (vmax - vmin) * 0.3 else "white",
                )

    return im


def _curve_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    metric_col: str,
    color: str,
    marker: str,
    label: str,
) -> None:
    grouped = data.groupby(x_col)[metric_col].mean().reset_index()
    ax.plot(grouped[x_col], grouped[metric_col], color=color, marker=marker, markersize=4, label=label)


def _find_q_single_param_csv() -> pd.DataFrame | None:
    candidates = sorted(ROOT.rglob("gate_parameter_sensitivity_results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    full = pd.read_csv(candidates[0])
    q_data = full[full["parameter"] == "threshold_quantile"].copy()
    if q_data.empty:
        return None
    return q_data


def render(output_path: Path | None = None) -> Path:
    out = Path(output_path or OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)

    xy_dtype = np.float64
    ratio_csv, ratio_data = _latest_csv("ratio")
    timing_csv, timing_data = _latest_csv("timescale")

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.45, wspace=0.35)

    ax_ratio = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    ax_timescale = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    ax_q = [fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3])]

    x_ratio, y_ratio = "x_value", "y_value"
    x_tscale, y_tscale = "x_value", "y_value"

    ratio_xlabel = PARAM_LABELS_ALPHA["threshold_bias"]
    ratio_ylabel = PARAM_LABELS_ALPHA["trigger_hysteresis_margin"]
    tscale_xlabel = PARAM_LABELS_ALPHA["local_window_size"]
    tscale_ylabel = PARAM_LABELS_ALPHA["score_short_weight"]

    for i, (tag, col, label) in enumerate(METRICS):
        cap = ratio_data[col].max() * 1.02
        im = _heatmap_panel(ax_ratio[i], ratio_data, x_ratio, y_ratio, col, f"(a){['i','ii','iii'][i]}",
                            f"{label} %", vmin=ratio_data[col].min(), vmax=cap)
        ax_ratio[i].set_xlabel(ratio_xlabel, fontsize=9)
        ax_ratio[i].set_ylabel(ratio_ylabel, fontsize=9)
        cb = fig.colorbar(im, ax=ax_ratio[i], shrink=0.85, aspect=20)
        cb.set_label(label, fontsize=8)

    for i, (tag, col, label) in enumerate(METRICS):
        cap = timing_data[col].max() * 1.02
        im = _heatmap_panel(ax_timescale[i], timing_data, x_tscale, y_tscale, col, f"(b){['i','ii','iii'][i]}",
                            f"{label} %", vmin=timing_data[col].min(), vmax=cap)
        ax_timescale[i].set_xlabel(tscale_xlabel, fontsize=9)
        ax_timescale[i].set_ylabel(tscale_ylabel, fontsize=9)
        cb = fig.colorbar(im, ax=ax_timescale[i], shrink=0.85, aspect=20)
        cb.set_label(label, fontsize=8)

    q_data = _find_q_single_param_csv()
    if q_data is not None:
        colors_q = ["#2563EB", "#DC2626", "#059669"]
        for i, (tag, col, label) in enumerate(METRICS):
            _curve_panel(ax_q[i], q_data, "candidate_value_numeric", col,
                         color=colors_q[i], marker="o", label=label)
            ax_q[i].set_xlabel(PARAM_LABELS_ALPHA["threshold_quantile"], fontsize=9)
            ax_q[i].set_ylabel(label, fontsize=9)
            ax_q[i].legend(fontsize=8)
    else:
        for i, ax in enumerate(ax_q):
            ax.text(0.5, 0.5, "(c) q 敏感\n请先运行\nsensitivity 实验",
                    transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#666")
            ax.set_title(f"(c){['i','ii','iii'][i]}", fontweight="bold", fontsize=11)
            if i == 0:
                continue
            ax_q[i].set_xlabel(r"$q$", fontsize=9)

    ax_q[0].set_title("(c)i", fontweight="bold", fontsize=11)
    ax_q[1].set_title("(c)ii/iii", fontweight="bold", fontsize=11)

    fig.suptitle("ET-PRL Gate Key Hyperparameter Sensitivity", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15, dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    try:
        render()
        print("Done  fig12 saved")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
