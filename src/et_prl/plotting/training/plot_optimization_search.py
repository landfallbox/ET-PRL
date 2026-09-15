from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from et_prl.config.base import project_root


def _running_best(values: pd.Series, minimize: bool) -> pd.Series:
    return values.cummin() if minimize else values.cummax()


def _load_gate_trials(file_path: Path) -> pd.DataFrame:
    trials = pd.read_csv(file_path)
    trials = trials[trials["state"] == "COMPLETE"].copy()
    trials = trials.sort_values("trial_number").reset_index(drop=True)
    trials["composite_score"] = pd.to_numeric(trials["composite_score"], errors="coerce")
    trials["running_best"] = _running_best(trials["composite_score"], minimize=False)
    return trials


def _plot_gate(trials: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5), dpi=300)
    scatter = plt.scatter(
        trials["trial_number"],
        trials["composite_score"],
        c=trials["action_rate"],
        cmap="viridis",
        s=18,
        alpha=0.8,
        label="Trial composite score",
    )
    plt.plot(trials["trial_number"], trials["running_best"], linewidth=2.0, label="Running best composite")
    best_row = trials.loc[trials["composite_score"].idxmax()]
    plt.scatter([best_row["trial_number"]], [best_row["composite_score"]], s=52, marker="*", zorder=5, label="Best trial")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Action Rate")
    plt.xlabel("Trial Number")
    plt.ylabel("Composite Score")
    plt.title("Gate Hyperparameter Search Process")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _latest_gate_trials_file() -> Path:
    """在 outputs/runs/online_anomaly_detection/optimization/<ts>/gate/ 下找最新的 gate_trials.csv。"""
    opt_root = project_root() / "outputs" / "runs" / "online_anomaly_detection" / "optimization"
    if not opt_root.exists():
        raise FileNotFoundError(f"未找到门控优化目录: {opt_root}")
    candidates = sorted(
        (p for p in opt_root.glob("*/gate/gate_trials.csv") if p.is_file()),
        key=lambda p: p.parent.parent.name,
    )
    if not candidates:
        raise FileNotFoundError(f"在 {opt_root} 下未找到 gate_trials.csv")
    return candidates[-1]


def main() -> None:
    gate_trials_file = _latest_gate_trials_file()

    pics_dir = project_root() / "outputs" / "figures"
    pics_dir.mkdir(parents=True, exist_ok=True)

    gate_trials = _load_gate_trials(gate_trials_file)

    _plot_gate(gate_trials, pics_dir / "fig4_4_2_gate_optimization_search.png")


if __name__ == "__main__":
    main()
