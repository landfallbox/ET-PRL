"""
基于 DQN 的 train/val 数据预热在线门控并保存状态
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.dqn_config import DQNConfig
from config.online_anomaly_detection_config import OnlineAnomalyDetectionConfig
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
    columns = OnlineAnomalyDetectionConfig.FEATURE_COLUMNS
    prewarm_data = _load_prewarm_data(columns)

    gate = StreamingAnomalyGate(
        feature_dim=len(columns),
        local_window_size=OnlineAnomalyDetectionConfig.GATE_LOCAL_WINDOW_SIZE,
        global_ema_decay=OnlineAnomalyDetectionConfig.GATE_GLOBAL_EMA_DECAY,
        reference_samples=OnlineAnomalyDetectionConfig.GATE_REFERENCE_SAMPLES,
        contamination=OnlineAnomalyDetectionConfig.GATE_CONTAMINATION,
        alpha_local_weight=OnlineAnomalyDetectionConfig.GATE_ALPHA_LOCAL_WEIGHT,
    )

    print(f"开始纯在线预热，样本数: {len(prewarm_data)}")

    trigger_count = 0
    anomaly_scores: list[float] = []

    for sample in prewarm_data:
        decision = gate.predict(sample)
        trigger_count += int(decision.gate_signal)
        anomaly_scores.append(float(decision.anomaly_score))

    gate.save_state(output_path)
    print(f"预热完成，状态已保存: {output_path}")

    stats = gate.get_statistics()
    print(f"样本计数: {stats['sample_count']}")
    print(f"当前阈值: {stats['threshold_optimizer']['adaptive_threshold']:.6f}")

    total_samples = len(prewarm_data)
    trigger_rate = trigger_count / total_samples if total_samples > 0 else 0.0
    print(f"触发率: {trigger_rate:.4%} ({trigger_count}/{total_samples})")

    if anomaly_scores:
        scores_np = np.asarray(anomaly_scores, dtype=np.float32)
        print(
            "分数分位数: "
            f"P50={np.quantile(scores_np, 0.50):.6f}, "
            f"P75={np.quantile(scores_np, 0.75):.6f}, "
            f"P90={np.quantile(scores_np, 0.90):.6f}, "
            f"P95={np.quantile(scores_np, 0.95):.6f}, "
            f"P99={np.quantile(scores_np, 0.99):.6f}"
        )

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
