"""门控超参优化的指标与数据切分。

本模块只含门控特征加载、召回类指标计算、验证集时间切分等纯计算函数，
不含优化主流程。供 gate_hyperparams_optimize.py 使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from et_prl.config.control_compare import ControlCompareConfig


def _load_prewarm_features(
    config: ControlCompareConfig,
    val_df_override: pd.DataFrame | None = None,
) -> np.ndarray:
    feature_columns = list(config.FEATURE_COLUMNS)

    train_df = pd.read_csv(config.get_train_data_path())
    val_df = val_df_override if val_df_override is not None else pd.read_csv(config.get_val_data_path())

    missing_train = [column for column in feature_columns if column not in train_df.columns]
    missing_val = [column for column in feature_columns if column not in val_df.columns]
    if missing_train:
        raise ValueError(f"训练集缺少门控特征列: {missing_train}")
    if missing_val:
        raise ValueError(f"验证集缺少门控特征列: {missing_val}")

    merged = pd.concat([train_df[feature_columns], val_df[feature_columns]], ignore_index=True)
    return merged.to_numpy(dtype=np.float32)


def _calculate_recall_metrics(step_results: pd.DataFrame) -> dict:
    violation = (
        (step_results["chiller_supply_temp"] < 15.0) | (step_results["chiller_supply_temp"] > 19.0)
    ).astype(int)
    gate_signal = step_results["gate_signal"].astype(int)

    tp = int(((gate_signal == 1) & (violation == 1)).sum())
    fp = int(((gate_signal == 1) & (violation == 0)).sum())
    fn = int(((gate_signal == 0) & (violation == 1)).sum())
    tn = int(((gate_signal == 0) & (violation == 0)).sum())

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    false_discovery_rate = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    action_rate = float(step_results["action_updated"].mean()) if not step_results.empty else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": recall,
        "precision": precision,
        "false_discovery_rate": false_discovery_rate,
        "false_positive_rate": false_positive_rate,
        "action_rate": action_rate,
    }


def _positive_part(value: float) -> float:
    return float(max(0.0, value))


def _split_validation_for_optimization(
    val_data: pd.DataFrame,
    fit_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None, float]:
    fit_ratio = float(np.clip(fit_ratio, 0.5, 0.95))

    if len(val_data) < 100:
        return val_data.reset_index(drop=True), None, 1.0

    split_index = int(len(val_data) * fit_ratio)
    split_index = int(np.clip(split_index, 1, len(val_data) - 1))

    fit_data = val_data.iloc[:split_index].reset_index(drop=True)
    holdout_data = val_data.iloc[split_index:].reset_index(drop=True)
    if holdout_data.empty:
        return fit_data, None, 1.0

    realized_ratio = float(len(fit_data) / max(len(val_data), 1))
    return fit_data, holdout_data, realized_ratio
