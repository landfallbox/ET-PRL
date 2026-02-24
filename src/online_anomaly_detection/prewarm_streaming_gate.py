"""
基于 DQN 的 train/val 数据预热在线门控并保存状态
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.dqn_config import DQNConfig
from src.online_anomaly_detection.streaming_anomaly_gate import StreamingAnomalyGate


def _load_prewarm_data(columns: list[str]) -> np.ndarray:
    train_path = DQNConfig.get_train_data_path()
    val_path = DQNConfig.get_val_data_path()

    if not train_path.exists():
        raise FileNotFoundError(f"训练集不存在: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"验证集不存在: {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    missing_train = [col for col in columns if col not in train_df.columns]
    missing_val = [col for col in columns if col not in val_df.columns]
    if missing_train:
        raise ValueError(f"训练集缺少列: {missing_train}")
    if missing_val:
        raise ValueError(f"验证集缺少列: {missing_val}")

    merged_df = pd.concat([train_df[columns], val_df[columns]], ignore_index=True)
    return merged_df.to_numpy(dtype=np.float32)


def prewarm_gate(output_path: Path) -> Path:
    columns = DQNConfig.STATE_COLUMNS
    prewarm_data = _load_prewarm_data(columns)

    gate = StreamingAnomalyGate(feature_dim=len(columns))

    print(f"开始预热门控，样本数: {len(prewarm_data)}")
    gate.initialize_with_data(prewarm_data)

    for sample in prewarm_data:
        gate.predict(sample)

    gate.save_state(output_path)
    print(f"预热完成，状态已保存: {output_path}")

    stats = gate.get_statistics()
    print(f"样本计数: {stats['sample_count']}")
    print(f"当前阈值: {stats['threshold_optimizer']['adaptive_threshold']:.6f}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="预热 StreamingAnomalyGate 并保存状态")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DQNConfig.get_data_dir() / "streaming_anomaly_gate_state.pkl"),
        help="状态文件输出路径",
    )
    args = parser.parse_args()

    prewarm_gate(Path(args.output))


if __name__ == "__main__":
    main()
