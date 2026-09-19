from __future__ import annotations

import numpy as np
import pandas as pd

from et_prl.agents.dqn import DQNAgent
from et_prl.environments import SequenceEnv
from et_prl.evaluation.control.common import (
    compute_extended_test_metrics,
    find_nearest_action_index,
)


def run_fixed_interval(
    agent: DQNAgent,
    env: SequenceEnv,
    action_space: np.ndarray,
    fixed_interval: int,
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
        should_update_action = steps % fixed_interval == 0
        action_reason = "fixed_interval" if should_update_action else "hold"

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
                "gate_signal": -1,
                "anomaly_score": np.nan,
                "adaptive_threshold": np.nan,
                "gate_confidence": np.nan,
            }
        )

        state = next_state
        if terminated:
            break

    avg_reward_per_env_step = total_reward / steps if steps > 0 else 0.0
    avg_reward_per_action = total_reward / action_update_count if action_update_count > 0 else 0.0
    extended_metrics = compute_extended_test_metrics(
        power_values=power_values,
        action_values=action_values,
        action_count=action_update_count,
    )

    summary = {
        "strategy": "fixed_interval",
        "steps": steps,
        "total_reward": float(total_reward),
        "avg_reward_per_action": float(avg_reward_per_action),
        # Backward-compatible alias; same value as avg_reward_per_action.
        "avg_reward_per_step": float(avg_reward_per_action),
        "avg_reward_per_env_step": float(avg_reward_per_env_step),
        "avg_energy_score": float(np.mean(energy_scores)) if energy_scores else 0.0,
        "avg_comfort_score": float(np.mean(comfort_scores)) if comfort_scores else 0.0,
        "avg_power_chiller": float(np.mean(power_values)) if power_values else 0.0,
        "action_count": int(action_update_count),
        "action_frequency": float(action_update_count / steps) if steps > 0 else 0.0,
        "event_trigger_count": 0,
        "event_trigger_rate": 0.0,
        "E_total_kwh": float(extended_metrics["E_total_kwh"]),
        "E_daily_kwh_per_day": float(extended_metrics["E_daily_kwh_per_day"]),
        "N_daily_count_per_day": float(extended_metrics["N_daily_count_per_day"]),
        "sigma_delta_a": float(extended_metrics["sigma_delta_a"]),
    }

    return summary, pd.DataFrame(records)
