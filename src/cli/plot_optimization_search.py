from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    gate_trials_file = (
        project_root
        / "logs"
        / "online_anomaly_detection"
        / "optimization"
        / "20260304_095858"
        / "gate"
        / "gate_trials.csv"
    )

    pics_dir = project_root / "docs" / "pics"
    pics_dir.mkdir(parents=True, exist_ok=True)

    gate_trials = _load_gate_trials(gate_trials_file)

    _plot_gate(gate_trials, pics_dir / "fig4_4_2_gate_optimization_search.png")


if __name__ == "__main__":
    main()
