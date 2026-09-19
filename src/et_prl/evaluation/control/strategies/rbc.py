from __future__ import annotations

import numpy as np
import pandas as pd

from et_prl.environments import SequenceEnv
from et_prl.evaluation.control.common import (
    compute_extended_test_metrics,
    find_nearest_action_index,
)


def run_rule_based_control(
    env: SequenceEnv,
    action_space: np.ndarray,
    fixed_setpoint: float = 7.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate a handcrafted rule-based control (RBC) baseline.

    Core logic uses a fixed chilled-water supply setpoint for the whole
    evaluation horizon. The controller does not adapt setpoint based on
    measured error, load, or weather.
    """
    action_min = float(np.min(action_space))
    action_max = float(np.max(action_space))
    fixed_setpoint_value = float(np.clip(float(fixed_setpoint), action_min, action_max))
    fixed_action_idx = int(find_nearest_action_index(action_space, fixed_setpoint_value))
    fixed_action_value = float(action_space[fixed_action_idx])

    _, _ = env.reset()
    steps = 0
    total_reward = 0.0
    power_values: list[float] = []
    energy_scores: list[float] = []
    comfort_scores: list[float] = []
    action_values: list[float] = []
    records: list[dict] = []
    action_update_count = 0

    measured_supply_temp = float(fixed_action_value)

    while True:
        current_action_idx = int(fixed_action_idx)
        current_action_value = float(fixed_action_value)
        target_setpoint = float(fixed_setpoint_value)
        action_updated = 1 if steps == 0 else 0
        action_reason = "rbc_fixed_setpoint" if action_updated == 1 else "rbc_fixed_setpoint_hold"

        action_update_count += int(action_updated)
        temp_error = float(measured_supply_temp - fixed_action_value)

        _, reward, terminated, _, info = env.step(current_action_value)

        steps += 1
        reward_value = float(reward)
        energy_score = float(info.get("energy_score", 0.0))
        comfort_score = float(info.get("comfort_score", 0.0))
        power_chiller = float(info.get("power_chiller", 0.0))
        chiller_supply_temp = float(info.get("chiller_supply_temp", 0.0))

        total_reward += reward_value
        power_values.append(power_chiller)
        energy_scores.append(energy_score)
        comfort_scores.append(comfort_score)
        action_values.append(current_action_value)
        measured_supply_temp = chiller_supply_temp

        records.append(
            {
                "step": steps,
                "action_idx": int(current_action_idx),
                "action_value": float(current_action_value),
                "selected_action_idx": int(current_action_idx),
                "action_updated": int(action_updated),
                "action_reason": action_reason,
                "reward": reward_value,
                "energy_score": energy_score,
                "comfort_score": comfort_score,
                "power_chiller": power_chiller,
                "chiller_supply_temp": chiller_supply_temp,
                "gate_signal": -1,
                "anomaly_score": np.nan,
                "adaptive_threshold": np.nan,
                "gate_confidence": np.nan,
                "rbc_target_setpoint": float(target_setpoint),
                "rbc_control_target": float(fixed_action_value),
                "rbc_temp_error": float(temp_error),
            }
        )

        if terminated:
            break

    extended_metrics = compute_extended_test_metrics(
        power_values=power_values,
        action_values=action_values,
        action_count=action_update_count,
    )

    summary = {
        "strategy": "rbc",
        "steps": steps,
        "total_reward": float(total_reward),
        "avg_reward_per_action": float(total_reward / action_update_count)
        if action_update_count > 0
        else 0.0,
        "avg_reward_per_step": float(total_reward / steps) if steps > 0 else 0.0,
        "avg_power_chiller": float(np.mean(power_values)) if power_values else 0.0,
        "avg_energy_score": float(np.mean(energy_scores)) if energy_scores else 0.0,
        "avg_comfort_score": float(np.mean(comfort_scores)) if comfort_scores else 0.0,
        "action_count": int(action_update_count),
        "E_total_kwh": float(extended_metrics["E_total_kwh"]),
        "E_daily_kwh_per_day": float(extended_metrics["E_daily_kwh_per_day"]),
        "N_daily_count_per_day": float(extended_metrics["N_daily_count_per_day"]),
        "sigma_delta_a": float(extended_metrics["sigma_delta_a"]),
        "rule_thresholds": {
            "source": "fixed_setpoint",
            "fixed_setpoint_requested": float(fixed_setpoint),
            "fixed_setpoint_applied": float(fixed_action_value),
        },
    }

    return summary, pd.DataFrame(records)
