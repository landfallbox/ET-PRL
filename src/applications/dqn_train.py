import argparse

from src.dqn.train_dqn import train_dqn


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN 训练入口")
    parser.add_argument("--runs", type=int, default=1, help="连续训练次数，默认 1")
    args = parser.parse_args()
    train_dqn(runs=args.runs)
