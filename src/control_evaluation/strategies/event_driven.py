from __future__ import annotations

import numpy as np
import pandas as pd
from ml_toolkit.rl import SequenceEnv

from src.control_evaluation.common import (
    compute_extended_evaluation_metrics,
    compute_violation_metrics,
    find_nearest_action_index,
)
from src.dqn.agent import DQNAgent
from src.online_anomaly_detection.streaming_anomaly_gate import StreamingAnomalyGate


def evaluate_event_driven(
    agent: DQNAgent,
    env: SequenceEnv,
    data: pd.DataFrame,
    action_space: np.ndarray,
    gate: StreamingAnomalyGate,
    feature_columns: list[str],
    supply_temp_ref: float,
) -> tuple[dict, pd.DataFrame]:
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    energy_scores: list[float] = []
    comfort_scores: list[float] = []
    power_values: list[float] = []
    supply_temperature_values: list[float] = []
    records: list[dict] = []
    action_update_count = 0
    action_values: list[float] = []

    current_action_idx = find_nearest_action_index(action_space, supply_temp_ref)
    current_action_value = float(action_space[current_action_idx])

    while True:
        feature_values = data.iloc[steps][feature_columns].to_numpy(dtype=np.float32)
        gate_decision = gate.predict(feature_values)

        gate_signal = int(gate_decision.gate_signal)
        anomaly_score = float(gate_decision.anomaly_score)
        adaptive_threshold = float(gate_decision.adaptive_threshold)
        decision_confidence = float(gate_decision.confidence)

        should_update_action = gate_signal == 1
        action_reason = "gate_trigger" if should_update_action else "hold"

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
        energy_scores.append(energy_score)
        comfort_scores.append(comfort_score)
        power_values.append(power_chiller)
        supply_temperature_values.append(chiller_supply_temp)
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
                "gate_signal": gate_signal,
                "anomaly_score": anomaly_score,
                "adaptive_threshold": adaptive_threshold,
                "gate_confidence": decision_confidence,
            }
        )

        state = next_state
        if terminated:
            break

    avg_reward_per_step = total_reward / steps if steps > 0 else 0.0
    violation_count, violation_time_pct = compute_violation_metrics(supply_temperature_values)
    gate_trigger_count = int(sum(int(record["gate_signal"] == 1) for record in records)) if records else 0
    gate_trigger_rate = gate_trigger_count / steps if steps > 0 else 0.0
    extended_metrics = compute_extended_evaluation_metrics(
        power_values=power_values,
        action_values=action_values,
        supply_temperature_values=supply_temperature_values,
        action_count=action_update_count,
        sample_interval_minutes=5.0,
        comfort_reference_temp=supply_temp_ref,
        severe_violation_delta=3.0,
    )

    summary = {
        "strategy": "event_driven",
        "steps": steps,
        "total_reward": float(total_reward),
        "avg_reward_per_step": float(avg_reward_per_step),
        "avg_energy_score": float(np.mean(energy_scores)) if energy_scores else 0.0,
        "avg_comfort_score": float(np.mean(comfort_scores)) if comfort_scores else 0.0,
        "avg_power_chiller": float(np.mean(power_values)) if power_values else 0.0,
        "action_count": int(action_update_count),
        "action_frequency": float(action_update_count / steps) if steps > 0 else 0.0,
        "temperature_violations": int(violation_count),
        "violation_time_pct": float(violation_time_pct),
        "event_trigger_count": int(gate_trigger_count),
        "event_trigger_rate": float(gate_trigger_rate),
        "E_total_kwh": float(extended_metrics["E_total_kwh"]),
        "E_daily_kwh_per_day": float(extended_metrics["E_daily_kwh_per_day"]),
        "N_daily_count_per_day": float(extended_metrics["N_daily_count_per_day"]),
        "sigma_delta_a": float(extended_metrics["sigma_delta_a"]),
        "severe_violation_count": int(extended_metrics["severe_violation_count"]),
        "severe_violation_rate_pct": float(extended_metrics["severe_violation_rate_pct"]),
        "severe_violation_monthly_mean": float(extended_metrics["severe_violation_monthly_mean"]),
    }

    return summary, pd.DataFrame(records)
