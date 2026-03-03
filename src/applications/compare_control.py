import argparse
from pathlib import Path

from config.compare_dqn_config import CompareDQNConfig
from src.comparison.compare_control_strategies import compare_control_strategies


def main() -> None:
    parser = argparse.ArgumentParser(description="控制策略性能对比")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--fixed_interval", type=int, default=1)
    parser.add_argument(
        "--gate_state_path",
        type=str,
        default=str(CompareDQNConfig.get_data_dir() / "streaming_anomaly_gate_state.pkl"),
    )
    args = parser.parse_args()

    compare_control_strategies(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )


if __name__ == "__main__":
    main()
