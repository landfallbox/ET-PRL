"""Generate the publication-quality gate parameter sensitivity figures.

The figures are split into a sensitivity-span heatmap for the overall ranking
and six normalized response panels for the detailed one-factor response of
each selected gate hyperparameter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

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


DEFAULT_RESULTS_ROOT = Path("logs/online_anomaly_detection/sensitivity")
DEFAULT_SUMMARY_OUTPUT_RELATIVE_PATH = Path("docs/pics/fig12_gate_key_parameter_sensitivity.svg")
DEFAULT_RESPONSE_OUTPUT_RELATIVE_PATH = Path(
    "docs/pics/fig13_gate_key_parameter_sensitivity_response_curves.svg"
)
RESULTS_FILENAME = "gate_parameter_sensitivity_results.csv"


@dataclass(frozen=True)
class ParameterPanel:
    name: str
    label: str
    panel_label: str


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    summary_label: str
    color: str


SELECTED_PANELS: tuple[ParameterPanel, ...] = (
    ParameterPanel("threshold_bias", r"$b_{\mathrm{bias}}$", "(a)"),
    ParameterPanel("trigger_hysteresis_margin", r"$m_{\mathrm{hys}}$", "(b)"),
    ParameterPanel("local_window_size", r"$W$", "(c)"),
    ParameterPanel("threshold_quantile", r"$q$", "(d)"),
    ParameterPanel("threshold_mad_scale", r"$\kappa$", "(e)"),
    ParameterPanel("score_short_weight", r"$w_{\mathrm{s}}$", "(f)"),
)


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="total_reward",
        label=r"$R_{\text{test}}$",
        summary_label=r"$R_{\text{test}}$",
        color="#1D4ED8",
    ),
    MetricSpec(
        name="action_frequency",
        label=r"$N_{\text{daily}}$",
        summary_label=r"$N_{\text{daily}}$",
        color="#D97706",
    ),
    MetricSpec(
        name="E_daily_kwh_per_day",
        label=r"$E_{\text{daily}}$",
        summary_label=r"$E_{\text{daily}}$",
        color="#047857",
    ),
)


REQUIRED_COLUMNS = {
    "parameter",
    "candidate_value_numeric",
    "is_baseline_value",
    "total_reward",
    "action_frequency",
    "E_daily_kwh_per_day",
}


class _SilentLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None


def _resolve_results_path(results_path: Path | None, repo_root: Path) -> Path:
    if results_path is not None:
        resolved = results_path.expanduser()
        if resolved.is_file():
            return resolved
        if resolved.is_dir():
            candidates = list(resolved.rglob(RESULTS_FILENAME))
            if candidates:
                return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))
            raise FileNotFoundError(f"No sensitivity results CSV found under directory: {resolved}")
        raise FileNotFoundError(f"Sensitivity results file not found: {resolved}")

    search_root = repo_root / DEFAULT_RESULTS_ROOT
    if not search_root.exists():
        raise FileNotFoundError(
            f"Sensitivity results directory not found: {search_root}. "
            "Run `uv run event-dqn-analyze-gate-sensitivity` first or pass `--results-path`."
        )

    candidates = list(search_root.rglob(RESULTS_FILENAME))
    if not candidates:
        raise FileNotFoundError(
            f"No sensitivity results CSV found under: {search_root}. "
            "Run `uv run event-dqn-analyze-gate-sensitivity` first or pass `--results-path`."
        )

    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def _resolve_output_paths(output_path: Path | None, repo_root: Path) -> tuple[Path, Path]:
    if output_path is None:
        return (
            repo_root / DEFAULT_SUMMARY_OUTPUT_RELATIVE_PATH,
            repo_root / DEFAULT_RESPONSE_OUTPUT_RELATIVE_PATH,
        )

    summary_output_path = output_path.expanduser()
    if not summary_output_path.suffix:
        summary_output_path = summary_output_path.with_suffix(".svg")

    response_output_path = summary_output_path.with_name(
        f"{summary_output_path.stem}_response_curves{summary_output_path.suffix}"
    )
    return summary_output_path, response_output_path


def _load_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"Sensitivity results file not found: {results_path}")

    results = pd.read_csv(results_path)
    missing = REQUIRED_COLUMNS.difference(results.columns)
    if missing:
        raise KeyError(f"Missing required columns in {results_path}: {sorted(missing)}")

    numeric_columns = ["candidate_value_numeric", *(metric.name for metric in METRICS)]
    for column in numeric_columns:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    results = results.dropna(subset=numeric_columns)
    if results.empty:
        raise ValueError(f"No valid sensitivity rows found in: {results_path}")

    return results


def _baseline_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["is_baseline_value"].astype(str).str.lower().isin({"true", "1", "yes"})


def _safe_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.zeros_like(values, dtype=float)

    finite_values = values[finite_mask]
    v_min = float(np.min(finite_values))
    v_max = float(np.max(finite_values))
    if np.isclose(v_min, v_max):
        scaled = np.full_like(values, 0.5, dtype=float)
        scaled[~finite_mask] = 0.0
        return scaled

    scaled = np.zeros_like(values, dtype=float)
    scaled[finite_mask] = (finite_values - v_min) / (v_max - v_min)
    return scaled


def _format_candidate_tick(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _panel_summary_frame(results: pd.DataFrame) -> pd.DataFrame:
    required_names = {panel.name for panel in SELECTED_PANELS}
    present_names = set(results["parameter"].astype(str))
    missing_names = required_names.difference(present_names)
    if missing_names:
        raise ValueError(f"Missing sensitivity rows for parameters: {sorted(missing_names)}")

    rows: list[dict[str, object]] = []
    for panel in SELECTED_PANELS:
        frame = results[results["parameter"] == panel.name].copy()
        if frame.empty:
            raise ValueError(f"No sensitivity rows found for parameter: {panel.name}")

        row: dict[str, object] = {
            "parameter": panel.name,
            "label": panel.label,
        }
        for metric in METRICS:
            span = float(frame[metric.name].max() - frame[metric.name].min())
            row[f"{metric.name}_span"] = span
        rows.append(row)

    summary = pd.DataFrame.from_records(rows)
    summary = summary.sort_values("total_reward_span", ascending=False, kind="mergesort").reset_index(drop=True)

    for metric in METRICS:
        summary[f"{metric.name}_span_norm"] = _safe_minmax(summary[f"{metric.name}_span"].to_numpy(dtype=float))

    return summary


def _plot_summary_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    value_columns = [f"{metric.name}_span_norm" for metric in METRICS]
    matrix = summary[value_columns].to_numpy(dtype=float)

    image = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([metric.summary_label for metric in METRICS], fontsize=9.4)
    for tick_label, metric in zip(ax.get_xticklabels(), METRICS, strict=False):
        tick_label.set_color(metric.color)
        tick_label.set_fontweight("semibold")

    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels(summary["label"].tolist(), fontsize=9.2)
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(len(summary)):
        for col_index in range(len(METRICS)):
            value = float(matrix[row_index, col_index])
            text_color = "white" if value >= 0.55 else "#111827"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=text_color,
                fontweight="semibold",
            )

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.024, pad=0.02)
    colorbar.set_label("Normalized span", fontsize=9.4)
    colorbar.ax.tick_params(labelsize=8.6)


def _plot_response_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    panel: ParameterPanel,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    data = frame[frame["parameter"] == panel.name].copy().sort_values("candidate_value_numeric", kind="mergesort")
    if data.empty:
        raise ValueError(f"No rows found for parameter: {panel.name}")

    x_values = data["candidate_value_numeric"].to_numpy(dtype=float)
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    x_span = max(x_max - x_min, 1e-9)
    x_pad = max(0.06 * x_span, 0.015 * max(1.0, abs(x_min), abs(x_max)))

    baseline_rows = data[_baseline_mask(data)]
    if not baseline_rows.empty:
        baseline_x = float(baseline_rows.iloc[0]["candidate_value_numeric"])
        ax.axvline(
            baseline_x,
            color="#9CA3AF",
            linestyle="--",
            linewidth=0.95,
            alpha=0.85,
            zorder=1,
        )

    for metric in METRICS:
        y_values = _safe_minmax(data[metric.name].to_numpy(dtype=float))
        ax.plot(
            x_values,
            y_values,
            color=metric.color,
            linewidth=1.65,
            marker="o",
            markersize=4.5,
            markerfacecolor=metric.color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.28, color="#BFC7D5")
    ax.set_title(f"{panel.panel_label}  {panel.label}", loc="left", fontsize=10.7, fontweight="bold", pad=4.5)
    ax.tick_params(axis="both", labelsize=8.7)
    ax.tick_params(axis="x", pad=1.8)

    ax.set_xticks(x_values)
    ax.set_xticklabels([_format_candidate_tick(value) for value in x_values], fontsize=8.0)

    if show_ylabel:
        ax.set_ylabel("Normalized response", fontsize=9.3)
    else:
        ax.tick_params(axis="y", labelleft=False)

    if show_xlabel:
        ax.set_xlabel("Candidate value", fontsize=9.3)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_format = output_path.suffix.lstrip(".") or "svg"
    fig.savefig(output_path, format=file_format, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_summary_figure(summary: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(12.0, 4.8), dpi=160, constrained_layout=False)
    ax = fig.add_subplot(1, 1, 1)
    _plot_summary_heatmap(ax, summary)
    fig.subplots_adjust(left=0.08, right=0.965, top=0.90, bottom=0.12)
    _save_figure(fig, output_path)


def _render_response_figure(results: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(13.8, 7.7), dpi=160, constrained_layout=False)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.48, wspace=0.28)

    response_axes: list[plt.Axes] = []
    for index, panel in enumerate(SELECTED_PANELS):
        row = index // 3
        col = index % 3
        sharey_ax = response_axes[0] if response_axes else None
        ax = fig.add_subplot(grid[row, col], sharey=sharey_ax)
        _plot_response_panel(
            ax,
            results,
            panel,
            show_ylabel=(col == 0),
            show_xlabel=(row == 1),
        )
        response_axes.append(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=metric.color,
            marker="o",
            markersize=5.6,
            linewidth=1.65,
            markerfacecolor=metric.color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=metric.label,
        )
        for metric in METRICS
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#9CA3AF",
            linestyle="--",
            linewidth=0.95,
            label="Selected value",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        fontsize=9.0,
        handlelength=1.8,
        columnspacing=1.6,
    )

    fig.subplots_adjust(left=0.055, right=0.965, top=0.91, bottom=0.08)
    _save_figure(fig, output_path)


def generate_figures(results_path: Path | None, output_path: Path | None) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    resolved_results_path = _resolve_results_path(results_path, repo_root)
    resolved_summary_output_path, resolved_response_output_path = _resolve_output_paths(output_path, repo_root)

    results = _load_results(resolved_results_path)
    summary = _panel_summary_frame(results)

    _render_summary_figure(summary, resolved_summary_output_path)
    _render_response_figure(results, resolved_response_output_path)

    print(f"Saved sensitivity summary figure: {resolved_summary_output_path}")
    print(f"Saved sensitivity response figure: {resolved_response_output_path}")
    print(f"Source results: {resolved_results_path}")


def generate_figure(results_path: Path | None, output_path: Path | None) -> None:
    generate_figures(results_path=results_path, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the publication-quality gate sensitivity figure.")
    parser.add_argument("--results-path", type=Path, default=None, help="Sensitivity results CSV or directory")
    parser.add_argument("--output-path", type=Path, default=None, help="Figure output path")
    args = parser.parse_args()

    generate_figures(results_path=args.results_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
