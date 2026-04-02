from __future__ import annotations

import numpy as np
import pandas as pd
from ml_toolkit.rl import SequenceEnv

from src.control_evaluation.common import compute_extended_test_metrics
from src.control_evaluation.strategies.dynamics_model import HVACDynamicsModel
from src.dqn.rewards import RewardCalculator


def test_mbc(
    env: SequenceEnv,
    action_space: np.ndarray,
    reward_calc: RewardCalculator,
    data: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate Model-Based Control (MBC) using complete HVAC dynamics model.
    
    Implements true Model-Based Control by:
    1. Learning complete environment dynamics from historical data
    2. Predicting full system state (power, supply temp, efficiency, comfort)
    3. Computing expected objective with energy-saving priority
    4. Greedily selecting action that maximizes predicted objective
    
    This is a complete dynamics-based controller, not simple energy predictor.
    """
    # 初始化HVAC系统动态模型
    dynamics_model = HVACDynamicsModel(
        data=data if data is not None else pd.DataFrame(),
        action_space=action_space,
        info_history=None,
        reward_calc=reward_calc,
        identification_sample_count=256,
    )
    
    state, _ = env.reset()
    total_objective = 0.0
    steps = 0
    power_values: list[float] = []
    records: list[dict] = []
    action_update_count = 0
    action_values: list[float] = []
    tracking_states: list[dict] = []
    
    # 初始化系统状态（为第一步预测做准备）
    current_action = float(action_space[len(action_space) // 2])
    current_power = 300.0  # kW
    current_supply_temp = 7.0  # ℃
    current_cooling_load = 3500.0  # kW
    current_outdoor_temp = 25.0  # ℃

    while True:
        # 用动态模型批量预测每个候选动作的结果
        predicted_objectives = dynamics_model.batch_predict(
            current_states=tracking_states if tracking_states else [
                {
                    "action": current_action,
                    "power_chiller": current_power,
                    "chiller_supply_temp": current_supply_temp,
                    "cooling_load": current_cooling_load,
                    "outdoor_temp": current_outdoor_temp,
                }
            ],
            action_candidates=action_space,
            reward_calc=reward_calc,
            current_step=steps,
        )
        
        # 选择预测奖励最高的动作
        selected_action_idx = int(np.argmax(predicted_objectives))
        current_action = float(action_space[selected_action_idx])
        predicted_best_objective = float(predicted_objectives[selected_action_idx])
        action_update_count += 1

        previous_action = float(tracking_states[-1]["action"]) if tracking_states else current_action
        previous_power = float(tracking_states[-1]["power_chiller"]) if tracking_states else current_power
        previous_supply_temp = float(tracking_states[-1]["chiller_supply_temp"]) if tracking_states else current_supply_temp

        next_state, reward, terminated, _, info = env.step(current_action)

        steps += 1
        energy_score = float(info.get("energy_score", 0.0))
        action_range = float(max(np.max(action_space) - np.min(action_space), 1e-6))
        action_delta_penalty = 0.005 * abs(current_action - previous_action) / action_range
        objective_value = float(energy_score - action_delta_penalty)
        comfort_score = float(info.get("comfort_score", 0.0))
        power_chiller = float(info.get("power_chiller", 0.0))
        chiller_supply_temp = float(info.get("chiller_supply_temp", 0.0))
        cooling_load = float(info.get("cooling_load", 0.0))
        outdoor_temp = float(info.get("outdoor_temp", 25.0))

        dynamics_model.update_with_observation(
            previous_action=previous_action,
            current_action=current_action,
            previous_power=previous_power,
            current_power=power_chiller,
            previous_temp=previous_supply_temp,
            current_temp=chiller_supply_temp,
        )

        # 更新跟踪状态用于下一步预测
        current_power = power_chiller
        current_supply_temp = chiller_supply_temp
        current_cooling_load = cooling_load
        current_outdoor_temp = outdoor_temp
        
        # 维护状态历史（最近保留20个时间步）
        tracking_states.append({
            "action": current_action,
            "power_chiller": power_chiller,
            "chiller_supply_temp": chiller_supply_temp,
            "cooling_load": cooling_load,
            "outdoor_temp": outdoor_temp,
            "energy_score": energy_score,
            "comfort_score": comfort_score,
        })
        if len(tracking_states) > 20:
            tracking_states.pop(0)

        total_objective += objective_value
        power_values.append(power_chiller)
        action_values.append(current_action)

        records.append(
            {
                "step": steps,
                "action_idx": int(selected_action_idx),
                "action_value": float(current_action),
                "selected_action_idx": int(selected_action_idx),
                "action_updated": 1,
                "action_reason": "model_based_dynamics_prediction",
                "objective": objective_value,
                "reward": float(reward),
                "predicted_objective": predicted_best_objective,
                "predicted_reward": predicted_best_objective,
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

    avg_objective_per_action = total_objective / action_update_count if action_update_count > 0 else 0.0
    extended_metrics = compute_extended_test_metrics(
        power_values=power_values,
        action_values=action_values,
        action_count=action_update_count,
        sample_interval_minutes=5.0,
    )

    summary = {
        "strategy": "mbc",
        "steps": steps,
        "total_objective": float(total_objective),
        "avg_objective_per_action": float(avg_objective_per_action),
        "avg_objective_per_step": float(avg_objective_per_action),
        "total_reward": float(total_objective),
        "avg_reward_per_action": float(avg_objective_per_action),
        "avg_reward_per_step": float(avg_objective_per_action),
        "avg_power_chiller": float(np.mean(power_values)) if power_values else 0.0,
        "avg_energy_score": float(np.mean([r["energy_score"] for r in records])) if records else 0.0,
        "action_count": int(action_update_count),
        "E_total_kwh": float(extended_metrics["E_total_kwh"]),
        "E_daily_kwh_per_day": float(extended_metrics["E_daily_kwh_per_day"]),
        "N_daily_count_per_day": float(extended_metrics["N_daily_count_per_day"]),
        "sigma_delta_a": float(extended_metrics["sigma_delta_a"]),
    }

    return summary, pd.DataFrame(records)
