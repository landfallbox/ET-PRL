import argparse

from src.dqn.optimize_dqn_hyperparams import optimize_dqn_hyperparameters


def main() -> None:
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
