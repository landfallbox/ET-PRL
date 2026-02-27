import argparse
from pathlib import Path

from src.dqn.eval_dqn import eval_dqn


def main() -> None:
    parser = argparse.ArgumentParser(description="固定间隔控制策略评估")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--fixed_interval", type=int, default=4)
    args = parser.parse_args()

    eval_dqn(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
    )
