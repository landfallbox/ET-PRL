"""Generate the publication-quality gate parameter sensitivity response figure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config.control_compare_config import ControlCompareConfig
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
    }
)


DEFAULT_RESULTS_ROOT = Path("logs/online_anomaly_detection/sensitivity")
DEFAULT_SUMMARY_OUTPUT_RELATIVE_PATH = Path("docs/pics/fig11_gate_key_parameter_sensitivity.svg")
DEFAULT_RESPONSE_OUTPUT_RELATIVE_PATH = Path(
    "docs/pics/fig12_gate_key_parameter_sensitivity_response_curves.svg"
)
RESULTS_FILENAME = "gate_parameter_sensitivity_results.csv"
TOP_RESPONSE_PANEL_COUNT: int | None = None


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
    source_name: str | None = None


PARAMETER_LABELS: dict[str, str] = {
    "threshold_bias": r"$b_{\mathrm{bias}}$",
    "trigger_hysteresis_margin": r"$m_{\mathrm{hys}}$",
    "local_window_size": r"$W$",
    "score_short_weight": r"$w_{\mathrm{s}}$",
    "threshold_quantile": r"$q$",
    "threshold_mad_scale": r"$\kappa$",
    "threshold_local_update_rate": r"$\lambda_{\mathrm{local}}$",
    "alpha_local_weight": r"$\alpha_{\mathrm{local}}$",
    "global_ema_decay": r"$\lambda_{\mathrm{global}}$",
    "score_medium_weight": r"$w_{\mathrm{m}}$",
    "threshold_quantile_weight": r"$\omega_q$",
}


PARAMETER_GROUPS: dict[str, str] = {
    "threshold_bias": "threshold",
    "threshold_quantile": "threshold",
    "threshold_mad_scale": "threshold",
    "threshold_local_update_rate": "threshold",
    "threshold_quantile_weight": "threshold",
    "trigger_hysteresis_margin": "threshold",
    "alpha_local_weight": "adaptive",
    "local_window_size": "statistical",
    "global_ema_decay": "statistical",
    "score_short_weight": "fusion",
    "score_medium_weight": "fusion",
}


GROUP_LABELS: dict[str, str] = {
    "threshold": "Threshold",
    "adaptive": "Adaptive",
    "statistical": "Statistical",
    "fusion": "Fusion",
}


GROUP_COLORS: dict[str, str] = {
    "threshold": "#2563EB",
    "adaptive": "#7C3AED",
    "statistical": "#0891B2",
    "fusion": "#059669",
}


PARAMETER_COLORS: dict[str, str] = {
    "threshold_bias": "#2563EB",
    "trigger_hysteresis_margin": "#DC2626",
    "local_window_size": "#0891B2",
    "score_short_weight": "#059669",
    "threshold_quantile": "#D97706",
    "threshold_mad_scale": "#7C3AED",
    "threshold_quantile_weight": "#4F46E5",
    "threshold_local_update_rate": "#EA580C",
    "score_medium_weight": "#0F766E",
    "global_ema_decay": "#64748B",
    "alpha_local_weight": "#A855F7",
}


PARAMETER_CONFIG_ATTRS: dict[str, str] = {
    "threshold_bias": "GATE_THRESHOLD_BIAS",
    "threshold_quantile": "THRESHOLD_QUANTILE",
    "threshold_mad_scale": "THRESHOLD_MAD_SCALE",
    "threshold_local_update_rate": "THRESHOLD_LOCAL_UPDATE_RATE",
    "threshold_quantile_weight": "THRESHOLD_QUANTILE_WEIGHT",
    "trigger_hysteresis_margin": "GATE_TRIGGER_HYSTERESIS_MARGIN",
    "alpha_local_weight": "GATE_ALPHA_LOCAL_WEIGHT",
    "local_window_size": "GATE_LOCAL_WINDOW_SIZE",
    "global_ema_decay": "GATE_GLOBAL_EMA_DECAY",
    "score_short_weight": "GATE_SCORE_SHORT_WEIGHT",
    "score_medium_weight": "GATE_SCORE_MEDIUM_WEIGHT",
}


DELTA_COLUMNS_BY_METRIC: dict[str, str] = {
    "total_reward": "delta_total_reward_vs_default_gate",
    "N_daily_count_per_day": "delta_N_daily_count_per_day_vs_default_gate",
    "E_daily_kwh_per_day": "delta_E_daily_kwh_per_day_vs_default_gate",
}


HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "et_prl_sensitivity",
    ["#F8FAFC", "#DBEAFE", "#FBBF24", "#DC2626"],
)
RESPONSE_CMAP = LinearSegmentedColormap.from_list(
    "et_prl_response",
    ["#2563EB", "#F8FAFC", "#DC2626"],
)
RESPONSE_HEATMAP_COLUMNS = 10
REPRESENTATIVE_PARAMETERS: tuple[str, ...] = (
    "threshold_bias",
    "local_window_size",
    "score_short_weight",
    "threshold_mad_scale",
)


RESPONSE_PARAMETER_DISPLAY_RANGES: dict[str, tuple[float, float]] = {
    "threshold_bias": (-0.05, 0.10),
    "local_window_size": (90.0, 240.0),
    "score_short_weight": (0.36, 0.72),
    "threshold_mad_scale": (0.80, 2.23),
}


RESPONSE_GRID_COLUMNS = 3


ABSOLUTE_RESPONSE_YLABELS: dict[str, str] = {
    "total_reward": "Absolute reward",
    "N_daily_count_per_day": "Daily action count",
    "E_daily_kwh_per_day": "Daily energy (kWh/day)",
}


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="total_reward",
        label=r"$R_{\text{test}}$",
        summary_label=r"$R_{\text{test}}$",
        color="#1D4ED8",
    ),
    MetricSpec(
        name="N_daily_count_per_day",
        label=r"$N_{\text{daily}}$",
        summary_label=r"$N_{\text{daily}}$",
        color="#D97706",
        source_name="action_frequency",
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

    results = _ensure_display_metric_columns(results)

    numeric_columns = [
        "candidate_value_numeric",
        *(metric.name for metric in METRICS),
        *(column for column in DELTA_COLUMNS_BY_METRIC.values() if column in results.columns),
    ]
    for column in numeric_columns:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    results = results.dropna(subset=numeric_columns)
    if results.empty:
        raise ValueError(f"No valid sensitivity rows found in: {results_path}")

    return results


def _baseline_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["is_baseline_value"].astype(str).str.lower().isin({"true", "1", "yes"})


def _infer_steps_per_day(results: pd.DataFrame) -> float:
    if "action_count" in results.columns and "action_frequency" in results.columns:
        action_count = pd.to_numeric(results["action_count"], errors="coerce")
        action_frequency = pd.to_numeric(results["action_frequency"], errors="coerce")
        valid = action_frequency > 0.0
        if valid.any():
            estimated_steps = action_count[valid] / action_frequency[valid]
            finite_steps = estimated_steps[np.isfinite(estimated_steps)]
            if not finite_steps.empty:
                steps = float(np.median(finite_steps))
                if steps > 0.0:
                    return 288.0
    return 288.0


def _ensure_display_metric_columns(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    if "N_daily_count_per_day" not in results.columns and "action_frequency" in results.columns:
        steps_per_day = _infer_steps_per_day(results)
        results["N_daily_count_per_day"] = pd.to_numeric(results["action_frequency"], errors="coerce") * steps_per_day

    if "delta_N_daily_count_per_day_vs_default_gate" not in results.columns:
        if "delta_action_frequency_vs_default_gate" in results.columns:
            steps_per_day = _infer_steps_per_day(results)
            results["delta_N_daily_count_per_day_vs_default_gate"] = (
                pd.to_numeric(results["delta_action_frequency_vs_default_gate"], errors="coerce") * steps_per_day
            )
        elif "N_daily_count_per_day" in results.columns:
            deltas = pd.Series(np.nan, index=results.index, dtype=float)
            for parameter_name, group in results.groupby("parameter", sort=False):
                baseline_rows = group[_baseline_mask(group)]
                if not baseline_rows.empty:
                    baseline_value = float(baseline_rows.iloc[0]["N_daily_count_per_day"])
                else:
                    baseline_value = float(group["N_daily_count_per_day"].median())
                deltas.loc[group.index] = pd.to_numeric(group["N_daily_count_per_day"], errors="coerce") - baseline_value
            results["delta_N_daily_count_per_day_vs_default_gate"] = deltas

    return results


def _configured_parameter_value(parameter_name: str) -> float | None:
    config_attr = PARAMETER_CONFIG_ATTRS.get(parameter_name)
    if config_attr is None or not hasattr(ControlCompareConfig, config_attr):
        return None
    try:
        return float(getattr(ControlCompareConfig, config_attr))
    except (TypeError, ValueError):
        return None


def _selected_parameter_value(frame: pd.DataFrame, parameter_name: str) -> float | None:
    baseline_rows = frame[_baseline_mask(frame)]
    if not baseline_rows.empty:
        return float(baseline_rows.iloc[0]["candidate_value_numeric"])
    return _configured_parameter_value(parameter_name)


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


def _safe_log_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    log_values = np.full_like(values, np.nan, dtype=float)
    log_values[finite_mask] = np.log1p(np.clip(values[finite_mask], a_min=0.0, a_max=None))
    return _safe_minmax(log_values)


def _safe_ratio_to_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    scaled = np.zeros_like(values, dtype=float)
    if not np.any(finite_mask):
        return scaled

    max_value = float(np.max(np.abs(values[finite_mask])))
    if np.isclose(max_value, 0.0):
        return scaled

    scaled[finite_mask] = np.clip(np.abs(values[finite_mask]) / max_value, 0.0, 1.0)
    return scaled


def _format_span_label(value: float) -> str:
    if not np.isfinite(value):
        return ""

    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    if magnitude >= 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if magnitude > 0:
        return "<0.01"
    return "0"


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


def _normalized_candidate_positions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    positions = np.zeros_like(values, dtype=float)
    if not np.any(finite_mask):
        return positions

    finite_values = values[finite_mask]
    x_min = float(np.min(finite_values))
    x_max = float(np.max(finite_values))
    if np.isclose(x_min, x_max):
        positions[finite_mask] = 0.5
        return positions

    positions[finite_mask] = (finite_values - x_min) / (x_max - x_min)
    return positions


def _normalized_selected_position(x_values: np.ndarray, selected_value: float | None) -> float | None:
    if selected_value is None:
        return None
    x_values = np.asarray(x_values, dtype=float)
    finite_values = x_values[np.isfinite(x_values)]
    if finite_values.size == 0:
        return None
    x_min = float(np.min(finite_values))
    x_max = float(np.max(finite_values))
    if np.isclose(x_min, x_max):
        return 0.5
    return float(np.clip((selected_value - x_min) / (x_max - x_min), 0.0, 1.0))


def _build_metric_response_matrix(
    results: pd.DataFrame,
    parameter_order: list[str],
    metric: MetricSpec,
    n_columns: int = RESPONSE_HEATMAP_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, n_columns)
    matrix = np.full((len(parameter_order), n_columns), np.nan, dtype=float)
    selected_positions = np.full(len(parameter_order), np.nan, dtype=float)

    for row_index, parameter_name in enumerate(parameter_order):
        data = results[results["parameter"] == parameter_name].copy().sort_values(
            "candidate_value_numeric",
            kind="mergesort",
        )
        if data.empty:
            continue

        x_values = data["candidate_value_numeric"].to_numpy(dtype=float)
        x_positions = _normalized_candidate_positions(x_values)
        y_values, is_relative = _metric_relative_change_percent(data, metric)
        if not is_relative:
            y_values = _safe_minmax(data[metric.name].to_numpy(dtype=float))

        order = np.argsort(x_positions, kind="mergesort")
        x_sorted = x_positions[order]
        y_sorted = y_values[order]
        finite_mask = np.isfinite(x_sorted) & np.isfinite(y_sorted)
        if not np.any(finite_mask):
            continue

        x_valid = x_sorted[finite_mask]
        y_valid = y_sorted[finite_mask]
        column_indices = np.clip(np.rint(x_valid * (n_columns - 1)).astype(int), 0, n_columns - 1)
        for column_index in np.unique(column_indices):
            matrix[row_index, column_index] = float(np.mean(y_valid[column_indices == column_index]))

        selected_position = _normalized_selected_position(
            x_values,
            _selected_parameter_value(data, parameter_name),
        )
        if selected_position is not None:
            selected_positions[row_index] = selected_position

    return matrix, selected_positions


def _metric_relative_change_percent(data: pd.DataFrame, metric: MetricSpec) -> tuple[np.ndarray, bool]:
    values = data[metric.name].to_numpy(dtype=float)
    delta_column = DELTA_COLUMNS_BY_METRIC.get(metric.name)
    if delta_column in data.columns:
        deltas = data[delta_column].to_numpy(dtype=float)
        baseline_values = values - deltas
        finite_baseline = baseline_values[np.isfinite(baseline_values)]
        if finite_baseline.size:
            baseline = float(np.median(finite_baseline))
            denominator = max(abs(baseline), 1e-8)
            return 100.0 * deltas / denominator, True

    baseline_rows = data[_baseline_mask(data)]
    if not baseline_rows.empty:
        baseline = float(baseline_rows.iloc[0][metric.name])
        denominator = max(abs(baseline), 1e-8)
        return 100.0 * (values - baseline) / denominator, True

    return _safe_minmax(values), False


def _panel_summary_frame(results: pd.DataFrame) -> pd.DataFrame:
    parameter_names = list(dict.fromkeys(results["parameter"].astype(str).tolist()))
    if not parameter_names:
        raise ValueError("No sensitivity parameters found in results")

    rows: list[dict[str, object]] = []
    for parameter_name in parameter_names:
        frame = results[results["parameter"] == parameter_name].copy()
        if frame.empty:
            raise ValueError(f"No sensitivity rows found for parameter: {parameter_name}")

        row: dict[str, object] = {
            "parameter": parameter_name,
            "label": PARAMETER_LABELS.get(parameter_name, parameter_name),
            "group": PARAMETER_GROUPS.get(parameter_name, "other"),
        }
        for metric in METRICS:
            span = float(frame[metric.name].max() - frame[metric.name].min())
            row[f"{metric.name}_span"] = span
        rows.append(row)

    summary = pd.DataFrame.from_records(rows)
    summary = summary.sort_values("total_reward_span", ascending=False, kind="mergesort").reset_index(drop=True)

    for metric in METRICS:
        summary[f"{metric.name}_span_display"] = _safe_log_minmax(
            summary[f"{metric.name}_span"].to_numpy(dtype=float)
        )
        summary[f"{metric.name}_span_score"] = _safe_ratio_to_max(
            summary[f"{metric.name}_span"].to_numpy(dtype=float)
        )
    score_columns = [f"{metric.name}_span_score" for metric in METRICS]
    summary["overall_span_score"] = summary[score_columns].mean(axis=1)

    return summary


def _build_response_panels(summary: pd.DataFrame, max_panels: int | None = TOP_RESPONSE_PANEL_COUNT) -> list[ParameterPanel]:
    panels: list[ParameterPanel] = []
    panel_source = summary if max_panels is None else summary.head(max_panels)
    for index, row in enumerate(panel_source.itertuples(index=False), start=0):
        panel_label = f"({chr(ord('a') + index)})"
        panels.append(
            ParameterPanel(
                name=str(row.parameter),
                label=str(row.label),
                panel_label=panel_label,
            )
        )
    return panels


def _plot_summary_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> plt.AxesImage:
    row_count = len(summary)
    y_positions = np.arange(row_count)
    score_matrix = np.column_stack([summary[f"{metric.name}_span_score"].to_numpy(dtype=float) for metric in METRICS])
    span_matrix = np.column_stack([summary[f"{metric.name}_span"].to_numpy(dtype=float) for metric in METRICS])
    image = ax.imshow(score_matrix, aspect="auto", cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0, zorder=1)

    for row_index, row in enumerate(summary.itertuples(index=False)):
        group = str(getattr(row, "group"))
        group_color = GROUP_COLORS.get(group, "#6B7280")
        ax.plot(
            [-0.64, -0.64],
            [row_index - 0.37, row_index + 0.37],
            color=group_color,
            linewidth=3.5,
            solid_capstyle="round",
            zorder=4,
        )
        for metric_index, _ in enumerate(METRICS):
            score = float(score_matrix[row_index, metric_index])
            span = float(span_matrix[row_index, metric_index])
            ax.text(
                metric_index,
                row_index,
                _format_span_label(span),
                ha="center",
                va="center",
                fontsize=8.0,
                color="white" if score >= 0.62 else "#111827",
                fontweight="semibold" if score >= 0.55 else "normal",
                zorder=3,
            )

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([metric.summary_label for metric in METRICS], fontsize=9.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["label"].tolist(), fontsize=9.1)
    ax.set_xlim(-0.76, len(METRICS) - 0.5)
    ax.set_ylim(row_count - 0.5, -0.5)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, row_count, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1, zorder=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Metric span heatmap", loc="left", fontsize=10.6, fontweight="bold", pad=11)
    return image


def _plot_summary_ranking(ax: plt.Axes, summary: pd.DataFrame) -> None:
    row_count = len(summary)
    y_positions = np.arange(row_count)
    reward_metric = METRICS[0]
    reward_scores = summary[f"{reward_metric.name}_span_score"].to_numpy(dtype=float)
    overall_scores = summary["overall_span_score"].to_numpy(dtype=float)

    ax.barh(
        y_positions,
        reward_scores,
        height=0.5,
        color=reward_metric.color,
        alpha=0.86,
        edgecolor="none",
        zorder=3,
    )
    ax.scatter(
        overall_scores,
        y_positions,
        s=32,
        color="#111827",
        edgecolor="white",
        linewidth=0.75,
        zorder=4,
        label="Overall",
    )

    for row_index, row in enumerate(summary.itertuples(index=False)):
        reward_score = float(getattr(row, f"{reward_metric.name}_span_score"))
        reward_span = float(getattr(row, f"{reward_metric.name}_span"))
        ax.text(
            min(reward_score + 0.025, 1.04),
            row_index,
            _format_span_label(reward_span),
            ha="left",
            va="center",
            fontsize=8.0,
            color="#111827",
            fontweight="semibold",
            zorder=5,
        )

    ax.set_ylim(row_count - 0.5, -0.5)
    ax.set_xlim(0.0, 1.13)
    ax.set_yticks(y_positions)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=8.6)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.65, alpha=0.30, color="#BFC7D5", zorder=0)
    ax.grid(False, axis="y")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_xlabel(r"Normalized $\Delta R_{\mathrm{test}}$ span", fontsize=9.2)
    ax.set_title("Reward ranking", loc="left", fontsize=10.6, fontweight="bold", pad=11)


def _add_group_legend(fig: plt.Figure, summary: pd.DataFrame, *, y: float) -> None:
    group_handles = [
        Line2D([0], [0], color=color, linewidth=3.5, label=GROUP_LABELS.get(group, group.title()))
        for group, color in GROUP_COLORS.items()
        if group in set(summary["group"].astype(str))
    ]
    if not group_handles:
        return
    fig.legend(
        handles=group_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=max(1, len(group_handles)),
        frameon=False,
        fontsize=8.5,
        handlelength=1.5,
        columnspacing=1.2,
    )


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
    selected_x = _selected_parameter_value(data, panel.name)
    x_for_limits = x_values if selected_x is None else np.concatenate([x_values, np.asarray([selected_x], dtype=float)])
    x_min = float(np.min(x_for_limits))
    x_max = float(np.max(x_for_limits))
    x_span = max(x_max - x_min, 1e-9)
    x_pad = max(0.06 * x_span, 0.015 * max(1.0, abs(x_min), abs(x_max)))

    if selected_x is not None:
        ax.axvline(
            selected_x,
            color="#64748B",
            linestyle="--",
            linewidth=1.05,
            alpha=0.88,
            zorder=1,
        )

    all_y_values: list[np.ndarray] = []
    all_relative = True
    for metric in METRICS:
        y_values, is_relative = _metric_relative_change_percent(data, metric)
        all_relative = all_relative and is_relative
        all_y_values.append(y_values)
        ax.plot(
            x_values,
            y_values,
            color=metric.color,
            linewidth=1.45,
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=1.05,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    if all_relative:
        finite_y = np.concatenate([values[np.isfinite(values)] for values in all_y_values if np.any(np.isfinite(values))])
        max_abs = float(np.max(np.abs(finite_y))) if finite_y.size else 1.0
        limit = max(1.0, max_abs * 1.16)
        ax.set_ylim(-limit, limit)
        ax.axhline(0.0, color="#CBD5E1", linewidth=0.8, zorder=1)
    else:
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.28, color="#BFC7D5")
    ax.set_title(f"{panel.panel_label}  {panel.label}", loc="left", fontsize=10.7, fontweight="bold", pad=4.5)
    ax.tick_params(axis="both", labelsize=8.7)
    ax.tick_params(axis="x", pad=1.8)

    ax.set_xticks(x_values)
    ax.set_xticklabels([_format_candidate_tick(value) for value in x_values], fontsize=8.0)

    if show_ylabel:
        ylabel = "Relative change from selected value (%)" if all_relative else "Normalized response"
        ax.set_ylabel(ylabel, fontsize=9.3)
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
    fig_height = max(5.3, 0.43 * len(summary) + 1.8)
    fig = plt.figure(figsize=(11.3, fig_height), dpi=160, constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[2.95, 0.16, 1.45], wspace=0.14)
    heatmap_ax = fig.add_subplot(grid[0, 0])
    colorbar_ax = fig.add_subplot(grid[0, 1])
    ranking_ax = fig.add_subplot(grid[0, 2])
    image = _plot_summary_heatmap(heatmap_ax, summary)
    _plot_summary_ranking(ranking_ax, summary)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Relative span score", fontsize=8.7)
    colorbar.ax.tick_params(labelsize=8.0, length=2.5)
    _add_group_legend(fig, summary, y=0.975)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.84, bottom=0.12)
    _save_figure(fig, output_path)


def _render_response_figure(summary: pd.DataFrame, results: pd.DataFrame, output_path: Path) -> None:
    available_parameters = set(results["parameter"].astype(str))
    parameter_order = [parameter for parameter in REPRESENTATIVE_PARAMETERS if parameter in available_parameters]
    if not parameter_order:
        parameter_order = summary["parameter"].astype(str).head(5).tolist()

    fig = plt.figure(figsize=(13.2, 4.95), dpi=160, constrained_layout=False)
    grid = fig.add_gridspec(1, len(METRICS), wspace=0.26)
    axes: list[plt.Axes] = []

    for metric_index, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[0, metric_index])
        axes.append(ax)
        metric_values: list[np.ndarray] = []

        for parameter_index, parameter_name in enumerate(parameter_order):
            data = results[results["parameter"] == parameter_name].copy().sort_values(
                "candidate_value_numeric",
                kind="mergesort",
            )
            display_range = RESPONSE_PARAMETER_DISPLAY_RANGES.get(parameter_name)
            if display_range is not None:
                lower_bound, upper_bound = display_range
                data = data[
                    data["candidate_value_numeric"].between(
                        lower_bound,
                        upper_bound,
                        inclusive="both",
                    )
                ]
            if data.empty:
                continue

            x_values = data["candidate_value_numeric"].to_numpy(dtype=float)
            x_positions = _normalized_candidate_positions(x_values)
            y_values = data[metric.name].to_numpy(dtype=float)
            metric_values.append(y_values)

            line_color = PARAMETER_COLORS.get(parameter_name, "#334155")
            ax.plot(
                x_positions,
                y_values,
                color=line_color,
                linewidth=1.8,
                marker="o",
                markersize=4.0,
                markerfacecolor="white",
                markeredgewidth=1.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3 + parameter_index,
            )

        finite_values = np.concatenate([values[np.isfinite(values)] for values in metric_values if np.any(np.isfinite(values))])
        if finite_values.size:
            y_min = float(np.min(finite_values))
            y_max = float(np.max(finite_values))
            y_span = y_max - y_min
            y_padding = max(y_span * 0.10, abs(y_max) * 0.02, 1.0)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.30, color="#BFC7D5", zorder=0)
        ax.set_title(f"({chr(ord('a') + metric_index)}) {metric.label}", loc="left", fontsize=11.3, fontweight="bold", pad=6)
        ax.set_xlabel("Normalized position within plotted range", fontsize=9.3)
        ax.set_ylabel(ABSOLUTE_RESPONSE_YLABELS.get(metric.name, "Absolute value"), fontsize=9.3)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="both", labelsize=8.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#9CA3AF")
        ax.spines["left"].set_color("#9CA3AF")
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)

    summary_by_parameter = summary.set_index("parameter", drop=False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=PARAMETER_COLORS.get(parameter_name, "#334155"),
            linewidth=1.8,
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            label=str(summary_by_parameter.loc[parameter_name, "label"]),
        )
        for parameter_name in parameter_order
        if parameter_name in summary_by_parameter.index
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=max(1, len(legend_handles)),
        frameon=False,
        fontsize=8.7,
        handlelength=1.7,
        columnspacing=1.25,
    )

    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.24)
    _save_figure(fig, output_path)


def generate_figures(results_path: Path | None, output_path: Path | None) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    resolved_results_path = _resolve_results_path(results_path, repo_root)
    _, resolved_response_output_path = _resolve_output_paths(output_path, repo_root)

    results = _load_results(resolved_results_path)
    summary = _panel_summary_frame(results)

    _render_response_figure(summary, results, resolved_response_output_path)

    print(f"Saved sensitivity response figure: {resolved_response_output_path}")
    print(f"Source results: {resolved_results_path}")


def generate_figure(results_path: Path | None, output_path: Path | None) -> None:
    generate_figures(results_path=results_path, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the publication-quality gate sensitivity response figure.")
    parser.add_argument("--results-path", type=Path, default=None, help="Sensitivity results CSV or directory")
    parser.add_argument("--output-path", type=Path, default=None, help="Figure output path")
    args = parser.parse_args()

    generate_figures(results_path=args.results_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
