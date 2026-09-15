import argparse
from pathlib import Path

from et_prl.config.loader import load_config
from et_prl.cli.overrides import apply_overrides, load_overrides
from et_prl.detection.prewarm import prewarm_gate


def main() -> None:
    dqn_config = load_config("dqn")
    gate_config = load_config("gate")

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
        default=str(dqn_config.get_data_dir() / "streaming_anomaly_gate_state.pkl"),
    )
    args = parser.parse_args()

    if args.gate_config_path:
        gate_config_path = Path(args.gate_config_path)
        gate_overrides = load_overrides(gate_config_path)
        gate_config = apply_overrides(gate_config, gate_overrides, source_name="GATE")

    prewarm_gate(Path(args.output), gate_config=gate_config)
