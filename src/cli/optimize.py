import argparse
from pathlib import Path

from src.cl_predict.lstm_hyperparams_optimize import optimize_lstm_hyperparameters
from src.dqn.dqn_hyperparams_optimize import optimize_dqn_hyperparameters
from src.online_anomaly_detection.gate_hyperparams_optimize import optimize_gate_hyperparameters


def main_dqn() -> None:
    parser = argparse.ArgumentParser(description="DQN 超参贝叶斯优化")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--max_episodes", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    optimize_dqn_hyperparameters(
        n_trials=int(args.n_trials),
        max_episodes=int(args.max_episodes),
        n_jobs=int(args.n_jobs),
    )


def main_lstm() -> None:
    optimize_lstm_hyperparameters()


def main_gate() -> None:
    parser = argparse.ArgumentParser(description="Gate 召回超参贝叶斯优化")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument(
        "--previous_phase_result_dir",
        type=str,
        default=None,
        help="前阶段结果目录（phase2/3用于继承前阶段最优参数），"
        "格式: logs/online_anomaly_detection/optimization/<timestamp>/phase1",
    )
    args = parser.parse_args()

    optimize_gate_hyperparameters(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        previous_phase_result_dir=Path(args.previous_phase_result_dir) if args.previous_phase_result_dir else None,
    )
