from __future__ import annotations

import numpy as np
import pandas as pd
from ml_toolkit.rl import SequenceEnv

from src.control_evaluation.common import compute_extended_evaluation_metrics, find_nearest_action_index
from src.dqn.agent import DQNAgent


def evaluate_static_threshold_etc(
    agent: DQNAgent,
    env: SequenceEnv,
    data: pd.DataFrame,
    action_space: np.ndarray,
    feature_columns: list[str],
    supply_temp_ref: float,
    cl_threshold: float,
    twb_threshold: float,
    cl_predict_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate ETC baseline with static expert thresholds on feature deltas."""
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    power_values: list[float] = []
    records: list[dict] = []
    action_update_count = 0
    action_values: list[float] = []

    current_action_idx = find_nearest_action_index(action_space, supply_temp_ref)
    current_action_value = float(action_space[current_action_idx])

    thresholds = {
        "CL": float(cl_threshold),
        "Twb": float(twb_threshold),
        "CL_predict": float(cl_predict_threshold),
    }
    prev_features: dict[str, float] | None = None

    while True:
        row = data.iloc[steps]
        current_features = {column: float(row[column]) for column in feature_columns if column in row.index}

        if prev_features is None:
            gate_signal = 1
            anomaly_score = 1.0
        else:
            ratios: list[float] = []
            for column, threshold in thresholds.items():
                if column not in current_features or column not in prev_features:
                    continue
                safe_threshold = max(float(threshold), 1e-9)
                delta = abs(current_features[column] - prev_features[column])
                ratios.append(float(delta / safe_threshold))
            anomaly_score = float(max(ratios)) if ratios else 0.0
            gate_signal = int(anomaly_score >= 1.0)

        should_update_action = gate_signal == 1
        action_reason = "static_threshold_trigger" if should_update_action else "hold"

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
                "adaptive_threshold": 1.0,
                "gate_confidence": float(min(1.0, anomaly_score)),
            }
        )

        prev_features = current_features
        state = next_state
        if terminated:
            break

    avg_reward_per_action = total_reward / action_update_count if action_update_count > 0 else 0.0
    extended_metrics = compute_extended_evaluation_metrics(
        power_values=power_values,
        action_values=action_values,
        action_count=action_update_count,
        sample_interval_minutes=5.0,
    )

    summary = {
        "strategy": "static_threshold_etc",
        "steps": steps,
        "total_reward": float(total_reward),
        "avg_reward_per_action": float(avg_reward_per_action),
        "avg_reward_per_step": float(avg_reward_per_action),
        "avg_power_chiller": float(np.mean(power_values)) if power_values else 0.0,
        "action_count": int(action_update_count),
        "E_total_kwh": float(extended_metrics["E_total_kwh"]),
        "E_daily_kwh_per_day": float(extended_metrics["E_daily_kwh_per_day"]),
        "N_daily_count_per_day": float(extended_metrics["N_daily_count_per_day"]),
        "sigma_delta_a": float(extended_metrics["sigma_delta_a"]),
    }

    return summary, pd.DataFrame(records)
