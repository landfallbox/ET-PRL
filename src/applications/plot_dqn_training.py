"""绘制 Fixed-step DQN 基准模型训练动态曲线（论文图）。"""

import argparse
from pathlib import Path

from src.dqn.plot_training import plot_dqn_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 DQN 训练曲线（论文配图）")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="训练实验目录路径（含 training_history.csv）；若不指定则自动选最新目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="图片输出目录；默认与 experiment_dir 相同",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=3,
        help="训练奖励平滑窗口（默认 3）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否在屏幕上显示图片",
    )
    args = parser.parse_args()

    plot_dqn_training_curves(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        smooth_window=args.smooth_window,
        show=args.show,
    )


if __name__ == "__main__":
    main()
