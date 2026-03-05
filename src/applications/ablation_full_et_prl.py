import argparse
from pathlib import Path

from src.ablation.full_et_prl import run_full_et_prl_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="消融实验：Full ET-PRL")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--fixed_interval", type=int, default=1)
    parser.add_argument(
        "--gate_state_path",
        type=str,
        default=None,
        help="门控状态文件路径；默认不加载（冷启动）",
    )
    args = parser.parse_args()

    run_full_et_prl_ablation(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )


if __name__ == "__main__":
    main()
