import argparse
from pathlib import Path

from src.cl_predict.eval_lstm import eval_lstm


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM 模型评估")
    parser.add_argument("--experiment_dir", type=str, default=None)
    args = parser.parse_args()

    eval_lstm(train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None)
