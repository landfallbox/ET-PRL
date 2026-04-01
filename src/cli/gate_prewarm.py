import argparse
from pathlib import Path

from config.dqn_config import DQNConfig
from config.online_anomaly_detection_config import OnlineAnomalyDetectionConfig
from src.cli.overrides import apply_overrides, load_overrides
from src.online_anomaly_detection.prewarm_streaming_gate import prewarm_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="预热 StreamingAnomalyGate 并保存状态")
    parser.add_argument(
        "--gate_config_path",
        type=str,
        default=None,
        help="门控超参配置 JSON 路径（支持包含 best_config_overrides 的优化输出）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DQNConfig.get_data_dir() / "streaming_anomaly_gate_state.pkl"),
    )
    args = parser.parse_args()

    if args.gate_config_path:
        gate_config_path = Path(args.gate_config_path)
        gate_overrides = load_overrides(gate_config_path)
        apply_overrides(OnlineAnomalyDetectionConfig, gate_overrides, source_name="GATE")

    prewarm_gate(Path(args.output))
