import argparse
from pathlib import Path

from src.cl_predict.lstm_hyperparams_optimize import optimize_lstm_hyperparameters
from src.dqn.dqn_hyperparams_optimize import optimize_dqn_hyperparameters
from src.online_anomaly_detection.gate_sensitivity_analysis import run_gate_parameter_sensitivity_analysis
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
    parser.add_argument(
        "--dqn_model_dir",
        type=str,
        default=None,
        help="DQN model run directory used to load checkpoints/best_model.pth; defaults to latest logs/dqn/train/<timestamp>",
    )
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
        train_experiment_dir=Path(args.dqn_model_dir) if args.dqn_model_dir else None,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        previous_phase_result_dir=Path(args.previous_phase_result_dir) if args.previous_phase_result_dir else None,
    )


def main_gate_sensitivity() -> None:
    parser = argparse.ArgumentParser(description="Gate 超参敏感性分析")
    parser.add_argument(
        "--dqn_model_dir",
        type=str,
        default=None,
        help="DQN model run directory used to load checkpoints/best_model.pth; defaults to latest logs/dqn/train/<timestamp>",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="用于分析的数据划分，默认 test",
    )
    parser.add_argument(
        "--parameters",
        type=str,
        nargs="+",
        default=None,
        help="要分析的参数名列表；默认分析论文中的关键门控参数",
    )
    args = parser.parse_args()

    run_gate_parameter_sensitivity_analysis(
        train_experiment_dir=Path(args.dqn_model_dir) if args.dqn_model_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        data_split=str(args.data_split),
        parameters=[str(param) for param in args.parameters] if args.parameters else None,
    )
