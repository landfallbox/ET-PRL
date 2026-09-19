from __future__ import annotations

import numpy as np
import pandas as pd

from et_prl.environments import SequenceEnv
from et_prl.evaluation.control.common import (
    compute_extended_test_metrics,
    find_nearest_action_index,
)


def run_pid(
    env: SequenceEnv,
    action_space: np.ndarray,
    supply_temp_ref: float,
    kp: float,
    ki: float,
    kd: float,
    integral_limit: float,
    error_deadband: float = 0.1,
    derivative_filter_alpha: float = 0.7,
    max_action_step: float = 2.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate traditional discrete PID baseline using supply temperature closed-loop feedback.

    参数：
        supply_temp_ref: 供水温度设定值（℃）
        error_deadband: 温度误差死区，降低稳态抖动
        derivative_filter_alpha: 微分低通滤波系数
        max_action_step: 单步最大动作变化幅度，抑制激进调节
    """
    state, _ = env.reset()
    steps = 0
    power_values: list[float] = []
    records: list[dict] = []
    action_update_count = 0
    action_values: list[float] = []

    # 初始化 PID 状态
    action_space_min = float(np.min(action_space))
    action_space_max = float(np.max(action_space))
    current_action_value = float((action_space_min + action_space_max) / 2.0)
    target_supply_temp = float(supply_temp_ref)
    measured_supply_temp = target_supply_temp

    integral = 0.0
    prev_error = 0.0
    filtered_derivative = 0.0

    while True:
        # 温度闭环：误差 = 目标供水温度 - 实测供水温度
        raw_error = float(target_supply_temp - measured_supply_temp)
        error = 0.0 if abs(raw_error) <= abs(error_deadband) else raw_error

        # 条件积分：仅在误差不过大时积分，减轻风up
        if abs(error) <= 1.0:
            integral = float(np.clip(integral + error, -abs(integral_limit), abs(integral_limit)))
        derivative_raw = float(error - prev_error) if steps > 0 else 0.0
        filtered_derivative = float(
            derivative_filter_alpha * filtered_derivative
            + (1.0 - derivative_filter_alpha) * derivative_raw
        )

        # PID 计算原始连续控制量，再映射到离散动作空间
        raw_action_value = float(
            current_action_value + kp * error + ki * integral + kd * filtered_derivative
        )
        bounded_action_value = float(
            np.clip(
                raw_action_value,
                current_action_value - abs(max_action_step),
                current_action_value + abs(max_action_step),
            )
        )
        selected_action_idx = find_nearest_action_index(action_space, bounded_action_value)
        current_action_value = float(action_space[selected_action_idx])

        action_update_count += 1

        _, _, terminated, _, info = env.step(current_action_value)

        steps += 1
        energy_score = float(info.get("energy_score", 0.0))
        comfort_score = float(info.get("comfort_score", 0.0))
        power_chiller = float(info.get("power_chiller", 0.0))
        chiller_supply_temp = float(info.get("chiller_supply_temp", 0.0))

        measured_supply_temp = chiller_supply_temp
        prev_error = error

        power_values.append(power_chiller)
        action_values.append(current_action_value)

        records.append(
            {
                "step": steps,
                "action_idx": int(selected_action_idx),
                "action_value": float(current_action_value),
                "selected_action_idx": int(selected_action_idx),
                "action_updated": 1,
                "action_reason": "pid_discrete",
                "pid_error": float(error),
                "pid_integral": float(integral),
                "pid_derivative": float(filtered_derivative),
                "pid_target_supply_temp": target_supply_temp,
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

        if terminated:
            break

    extended_metrics = compute_extended_test_metrics(
        power_values=power_values,
        action_values=action_values,
        action_count=action_update_count,
    )

    summary = {
        "strategy": "pid",
        "steps": steps,
        "avg_power_chiller": float(np.mean(power_values)) if power_values else 0.0,
        "avg_energy_score": float(np.mean([r["energy_score"] for r in records]))
        if records
        else 0.0,
        "action_count": int(action_update_count),
        "E_total_kwh": float(extended_metrics["E_total_kwh"]),
        "E_daily_kwh_per_day": float(extended_metrics["E_daily_kwh_per_day"]),
        "N_daily_count_per_day": float(extended_metrics["N_daily_count_per_day"]),
        "sigma_delta_a": float(extended_metrics["sigma_delta_a"]),
    }

    return summary, pd.DataFrame(records)
