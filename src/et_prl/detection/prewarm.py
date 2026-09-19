"""
基于 DQN 的 train/val 数据预热在线门控并保存状态
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from et_prl.config.loader import get_default
from et_prl.data import load_prewarm_features
from et_prl.detection.streaming_gate import StreamingAnomalyGate


def prewarm_gate(output_path: Path, gate_config=None) -> Path:
    if gate_config is None:
        gate_config = get_default("gate")
    # 预热数据取自 dqn 数据目录（train+val），特征列取自 gate 配置
    columns = list(gate_config.FEATURE_COLUMNS)
    prewarm_data = load_prewarm_features(
        get_default("dqn"),
        feature_columns=columns,
    )

    gate = StreamingAnomalyGate(
        feature_dim=len(columns),
        local_window_size=gate_config.GATE_LOCAL_WINDOW_SIZE,
        global_ema_decay=gate_config.GATE_GLOBAL_EMA_DECAY,
        reference_samples=gate_config.GATE_REFERENCE_SAMPLES,
        alpha_local_weight=gate_config.GATE_ALPHA_LOCAL_WEIGHT,
        threshold_bias=gate_config.GATE_THRESHOLD_BIAS,
        threshold_quantile=gate_config.THRESHOLD_QUANTILE,
        threshold_mad_scale=gate_config.THRESHOLD_MAD_SCALE,
        threshold_local_update_rate=gate_config.THRESHOLD_LOCAL_UPDATE_RATE,
        threshold_quantile_weight=gate_config.THRESHOLD_QUANTILE_WEIGHT,
        threshold_min_samples_for_optimization=gate_config.THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION,
        score_short_weight=gate_config.GATE_SCORE_SHORT_WEIGHT,
        score_medium_weight=gate_config.GATE_SCORE_MEDIUM_WEIGHT,
        score_long_weight=gate_config.GATE_SCORE_LONG_WEIGHT,
        trigger_hysteresis_margin=gate_config.GATE_TRIGGER_HYSTERESIS_MARGIN,
        min_trigger_interval=gate_config.GATE_MIN_TRIGGER_INTERVAL,
        stats_ema_decay=gate_config.STATS_EMA_DECAY,
        stats_window_size=gate_config.STATS_WINDOW_SIZE,
        isolation_update_freq=gate_config.ISOLATION_UPDATE_FREQ,
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
