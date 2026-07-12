from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from ml_toolkit.rl import SequenceEnv

from src.control_evaluation.common import compute_extended_test_metrics, find_nearest_action_index
from src.dqn.agent import DQNAgent


def _robust_threshold(values: deque[float], fallback: float, mad_scale: float = 1.0) -> float:
    if len(values) < 2:
        return float(fallback)

    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    threshold = median + mad_scale * mad
    # Do not force threshold above fallback once we have enough data; this keeps triggers responsive.
    return float(max(1e-9, threshold))


def _initial_feature_thresholds(
    data: pd.DataFrame,
    feature_columns: list[str],
    min_thresholds: dict[str, float],
    warmup_window: int,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    if len(data) < 2:
        return {column: float(min_thresholds.get(column, 1e-6)) for column in feature_columns}

    sample_end = min(len(data), max(2, warmup_window + 1))
    warmup = data.iloc[:sample_end]

    for column in feature_columns:
        if column not in warmup.columns:
            thresholds[column] = float(min_thresholds.get(column, 1e-6))
            continue

        deltas = np.abs(np.diff(warmup[column].to_numpy(dtype=np.float64)))
        if deltas.size == 0:
            thresholds[column] = float(min_thresholds.get(column, 1e-6))
            continue

        robust = float(np.quantile(deltas, 0.60))
        seed = float(min_thresholds.get(column, robust))
        # Use the smaller one to avoid overly conservative initial thresholds.
        thresholds[column] = float(max(1e-9, min(seed, robust)))

    return thresholds


def test_event_triggered_etc(
    agent: DQNAgent,
    env: SequenceEnv,
    data: pd.DataFrame,
    action_space: np.ndarray,
    feature_columns: list[str],
    supply_temp_ref: float,
    min_delta_thresholds: dict[str, float] | None = None,
    threshold_window_size: int = 20,
    threshold_min_samples: int = 3,
    trigger_score_threshold: float = 0.6,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate an event-triggered ETC baseline driven by feature-change events.

    The method uses the current data columns (default: CL, Twb, CL_predict) and
    triggers an action update when any feature's normalized change exceeds
    a relatively sensitive trigger threshold.
    """
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    power_values: list[float] = []
    records: list[dict] = []
    action_update_count = 0
    action_values: list[float] = []

    current_action_idx = find_nearest_action_index(action_space, supply_temp_ref)
    current_action_value = float(action_space[current_action_idx])

    min_thresholds = min_delta_thresholds or {}
    adaptive_thresholds = _initial_feature_thresholds(
        data=data,
        feature_columns=feature_columns,
        min_thresholds=min_thresholds,
        warmup_window=threshold_window_size,
    )
    delta_buffers: dict[str, deque[float]] = {
        column: deque(maxlen=max(2, threshold_window_size)) for column in feature_columns
    }
    prev_features: dict[str, float] | None = None

    while True:
        row = data.iloc[steps]
        current_features = {column: float(row[column]) for column in feature_columns if column in row.index}

        if prev_features is None:
            gate_signal = 1
            anomaly_score = 1.0
            triggered_events = ["initialization"]
        else:
            event_scores: dict[str, float] = {}
            triggered_events = []
            for column in feature_columns:
                if column not in current_features or column not in prev_features:
                    continue

                delta = abs(current_features[column] - prev_features[column])
                delta_buffers[column].append(delta)
                threshold = _robust_threshold(
                    delta_buffers[column],
                    float(max(min_thresholds.get(column, 1e-6), adaptive_thresholds.get(column, 1e-6))),
                )
                adaptive_thresholds[column] = threshold
                event_score = float(delta / max(threshold, 1e-9))
                event_scores[column] = event_score

                if event_score >= float(trigger_score_threshold):
                    triggered_events.append(column)

            if len(delta_buffers[feature_columns[0]]) < threshold_min_samples:
                # 早期样本不足时，使用较保守的初始化阈值，避免阈值过快收缩
                gate_signal = 0 if not triggered_events else 1
            else:
                gate_signal = int(bool(triggered_events))

            anomaly_score = float(max(event_scores.values())) if event_scores else 0.0

        should_update_action = gate_signal == 1
        action_reason = "event_triggered_update" if should_update_action else "hold"

        selected_action_idx = -1
        if should_update_action:
            selected_action_idx = int(agent.select_action(state, training=False))
            current_action_idx = selected_action_idx
            current_action_value = float(agent.get_action_value(selected_action_idx))
            action_update_count += 1

        next_state, reward, terminated, _, info = env.step(current_action_value)

        steps += 1
        reward_value = float(reward)
        energy_score = float(info.get("energy_score", 0.0))
        comfort_score = float(info.get("comfort_score", 0.0))
        power_chiller = float(info.get("power_chiller", 0.0))
        chiller_supply_temp = float(info.get("chiller_supply_temp", 0.0))

        total_reward += reward_value
        power_values.append(power_chiller)
        action_values.append(current_action_value)

        records.append(
            {
                "step": steps,
                "action_idx": int(current_action_idx),
                "action_value": float(current_action_value),
                "selected_action_idx": int(selected_action_idx),
                "action_updated": int(should_update_action),
                "action_reason": action_reason,
                "reward": reward_value,
                "energy_score": energy_score,
                "comfort_score": comfort_score,
                "power_chiller": power_chiller,
                "chiller_supply_temp": chiller_supply_temp,
                "gate_signal": int(gate_signal),
                "anomaly_score": float(anomaly_score),
                "adaptive_threshold": float(max(adaptive_thresholds.values())) if adaptive_thresholds else 1.0,
                "gate_confidence": float(min(1.0, anomaly_score)),
                "triggered_events": ",".join(triggered_events),
            }
        )

        prev_features = current_features
        state = next_state
        if terminated:
            break

    avg_reward_per_action = total_reward / action_update_count if action_update_count > 0 else 0.0
    extended_metrics = compute_extended_test_metrics(
        power_values=power_values,
        action_values=action_values,
        action_count=action_update_count,
        sample_interval_minutes=5.0,
    )

    summary = {
        "strategy": "event_triggered_etc",
        "steps": steps,
        "total_reward": float(total_reward),
        "avg_reward_per_action": float(avg_reward_per_action),
        "avg_reward_per_step": float(avg_reward_per_action),
        "avg_power_chiller": float(np.mean(power_values)) if power_values else 0.0,
        "action_count": int(action_update_count),
        "event_trigger_count": int(action_update_count),
        "event_trigger_rate": float(action_update_count / steps) if steps > 0 else 0.0,
        "E_total_kwh": float(extended_metrics["E_total_kwh"]),
        "E_daily_kwh_per_day": float(extended_metrics["E_daily_kwh_per_day"]),
        "N_daily_count_per_day": float(extended_metrics["N_daily_count_per_day"]),
        "sigma_delta_a": float(extended_metrics["sigma_delta_a"]),
    }

    return summary, pd.DataFrame(records)