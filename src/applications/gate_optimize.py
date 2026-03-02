import argparse
from pathlib import Path

from src.online_anomaly_detection.optimize_gate_hyperparams import optimize_gate_hyperparameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 召回超参贝叶斯优化")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=None)
    args = parser.parse_args()

    optimize_gate_hyperparameters(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )
