import argparse
from pathlib import Path

from config.dqn_config import DQNConfig
from src.online_anomaly_detection.prewarm_streaming_gate import prewarm_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="预热 StreamingAnomalyGate 并保存状态")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DQNConfig.get_data_dir() / "streaming_anomaly_gate_state.pkl"),
    )
    args = parser.parse_args()

    prewarm_gate(Path(args.output))
