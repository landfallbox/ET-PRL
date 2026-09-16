import argparse
from pathlib import Path

from et_prl.training.lstm_hyperparams_optimize import optimize_lstm_hyperparameters
from et_prl.training.dqn_hyperparams_optimize import optimize_dqn_hyperparameters
from et_prl.experiments.gate_sensitivity import run_gate_mixed_sensitivity_analysis
from et_prl.training.gate_hyperparams_optimize import optimize_gate_hyperparameters


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
    parser = argparse.ArgumentParser(description="LSTM 超参贝叶斯优化")
    parser.add_argument("--n_trials", type=int, default=80)
    args = parser.parse_args()

    optimize_lstm_hyperparameters(n_trials=int(args.n_trials))


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
    parser = argparse.ArgumentParser(description="Gate 轻量级混合敏感性分析")
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
        "--bootstrap_samples",
        type=int,
        default=None,
        help="配对块 bootstrap 重采样次数；默认使用配置值",
    )
    parser.add_argument(
        "--bootstrap_block_size",
        type=int,
        default=None,
        help="配对块 bootstrap 的块长度；默认使用配置值",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=None,
        help="配对块 bootstrap 随机种子；默认使用配置值",
    )
    args = parser.parse_args()

    run_gate_mixed_sensitivity_analysis(
        train_experiment_dir=Path(args.dqn_model_dir) if args.dqn_model_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        data_split=str(args.data_split),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block_size=args.bootstrap_block_size,
        bootstrap_seed=args.bootstrap_seed,
    )
