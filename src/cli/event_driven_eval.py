import argparse
from pathlib import Path

from config.event_driven_dqn_config import EventDrivenDQNConfig
from src.online_anomaly_detection.eval_event_driven import eval_event_driven


def main() -> None:
    parser = argparse.ArgumentParser(description="事件驱动控制策略评估")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument(
        "--gate_state_path",
        type=str,
        default=str(EventDrivenDQNConfig.get_data_dir() / "streaming_anomaly_gate_state.pkl"),
    )
    args = parser.parse_args()

    eval_event_driven(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )
