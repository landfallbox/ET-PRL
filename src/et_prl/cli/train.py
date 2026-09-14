import argparse

from et_prl.training.lstm_train import train_lstm
from et_prl.training.dqn_train import train_dqn


def main_dqn() -> None:
    parser = argparse.ArgumentParser(description="DQN 训练入口")
    parser.add_argument("--runs", type=int, default=1, help="连续训练次数，默认 1")
    args = parser.parse_args()
    train_dqn(runs=args.runs)


def main_lstm() -> None:
    train_lstm()
