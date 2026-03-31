from __future__ import annotations

import numpy as np
import pandas as pd
from ml_toolkit.rl import SequenceEnv

from src.control_evaluation.common import compute_extended_evaluation_metrics, find_nearest_action_index


def evaluate_pid(
    env: SequenceEnv,
    action_space: np.ndarray,
    supply_temp_ref: float,
    kp: float,
    ki: float,
    kd: float,
    integral_limit: float,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate PID baseline using supply temperature as feedback signal."""
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    power_values: list[float] = []
    records: list[dict] = []
    action_update_count = 0
    action_values: list[float] = []

    current_action_idx = find_nearest_action_index(action_space, supply_temp_ref)
    current_action_value = float(action_space[current_action_idx])
    measured_supply_temp = float(supply_temp_ref)

    integral = 0.0
    prev_error = 0.0

    while True:
        error = float(supply_temp_ref - measured_supply_temp)
        integral = float(np.clip(integral + error, -abs(integral_limit), abs(integral_limit)))
        derivative = float(error - prev_error) if steps > 0 else 0.0

        raw_action_value = float(current_action_value + kp * error + ki * integral + kd * derivative)
        selected_action_idx = find_nearest_action_index(action_space, raw_action_value)
        current_action_idx = int(selected_action_idx)
        current_action_value = float(action_space[current_action_idx])
        action_update_count += 1

        next_state, reward, terminated, _, info = env.step(current_action_value)

        steps += 1
        reward_value = float(reward)
        energy_score = float(info.get("energy_score", 0.0))
        comfort_score = float(info.get("comfort_score", 0.0))
        power_chiller = float(info.get("power_chiller", 0.0))
        chiller_supply_temp = float(info.get("chiller_supply_temp", 0.0))

        measured_supply_temp = chiller_supply_temp
        prev_error = error

        total_reward += reward_value
        power_values.append(power_chiller)
        action_values.append(current_action_value)

        records.append(
            {
                "step": steps,
                "action_idx": int(current_action_idx),
                "action_value": float(current_action_value),
                "selected_action_idx": int(selected_action_idx),
                "action_updated": 1,
                "action_reason": "pid",
                "reward": reward_value,
                "pid_error": float(error),
                "pid_integral": float(integral),
                "pid_derivative": float(derivative),
                "energy_score": energy_score,
                "comfort_score": comfort_score,
                "power_chiller": power_chiller,
                "chiller_supply_temp": chiller_supply_temp,
                "gate_signal": -1,
                "anomaly_score": np.nan,
                "adaptive_threshold": np.nan,
                "gate_confidence": np.nan,
            }
        )

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
        "strategy": "pid",
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
