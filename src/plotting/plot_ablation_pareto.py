"""Generate Figure 5-2-7: ablation Pareto scatter.

Usage:
    uv run python src/plotting/plot_ablation_pareto.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
    }
)


PROPOSED_METHOD = "Full ET-PRL"


def _build_ablation_dataframe() -> pd.DataFrame:
    records = [
        {"method": "Local Threshold Only", "E_daily": 7367.57, "PPR": 95.30, "ACR": 1.9513},
        {"method": "Global Threshold Only", "E_daily": 7409.42, "PPR": 96.25, "ACR": 1.9565},
        {"method": "Short-scale Only", "E_daily": 7407.03, "PPR": 94.99, "ACR": 2.0754},
        {"method": "Medium-scale Only", "E_daily": 7484.70, "PPR": 99.38, "ACR": 1.1318},
        {"method": "Long-scale Only", "E_daily": 7431.26, "PPR": 98.90, "ACR": 1.0162},
        {"method": PROPOSED_METHOD, "E_daily": 7331.85, "PPR": 94.41, "ACR": 2.1525},
    ]
    return pd.DataFrame.from_records(records)


def _compute_pareto_mask(energy: np.ndarray, ppr: np.ndarray) -> np.ndarray:
    """Return non-dominated mask under min-energy and max-PPR objectives."""
    n = len(energy)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal = (energy[j] <= energy[i]) and (ppr[j] >= ppr[i])
            strictly_better = (energy[j] < energy[i]) or (ppr[j] > ppr[i])
            if better_or_equal and strictly_better:
                mask[i] = False
                break
    return mask


def _extract_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Pareto-optimal points and order them as the upper envelope."""
    pareto_mask = _compute_pareto_mask(
        df["E_daily"].to_numpy(dtype=float),
        df["PPR"].to_numpy(dtype=float),
    )
    pareto_df = df.loc[pareto_mask].copy()
    pareto_df = pareto_df.sort_values(by=["E_daily", "PPR"], ascending=[True, False], kind="mergesort")

    # Remove any residual numerical ties that would create non-front segments.
    frontier_rows: list[pd.Series] = []
    best_ppr = -np.inf
    tol = 1e-9
    for _, row in pareto_df.iterrows():
        if float(row["PPR"]) > best_ppr + tol:
            frontier_rows.append(row)
            best_ppr = float(row["PPR"])

    return pd.DataFrame(frontier_rows)


def generate_figure(output_path: Path) -> None:
    df = _build_ablation_dataframe()

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

    root = Path(__file__).parent.parent.parent
    output_path = args.output_path or (root / "docs" / "pics" / "fig10_ablation_pareto_scatter.svg")
    generate_figure(output_path=output_path)


if __name__ == "__main__":
    main()
