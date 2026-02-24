from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from ml_toolkit.rl import RewardCalculator as BaseRewardCalculator


class RewardCalculator(BaseRewardCalculator):
    def __init__(
        self,
        data: pd.DataFrame,
        action_space: np.ndarray,
        weight_efficiency: float,
        weight_comfort: float,
        target_supply_temp: float,
        coeff_path: Path,
        chiller_capacity: float,
        chiller_ref_power: float,
        supply_temp_ref: float,
        comfort_sigma: float,
        chiller_high_threshold: float,
        chiller_medium_threshold: float,
        chiller_low_threshold: float,
        f_nominal: float,
        f_cw: float,
        f_tower: float,
        f_chw: float,
        c_p: float,
        density_water: float,
    ) -> None:
        self.data = data
        self.action_space = np.asarray(action_space, dtype=np.float32)
        self.weight_efficiency = weight_efficiency
        self.weight_comfort = weight_comfort
        self.target_supply_temp = target_supply_temp

        coeff_path = Path(coeff_path)
        if not coeff_path.exists():
            raise FileNotFoundError(f"冷却塔系数文件不存在: {coeff_path}")
        self.coeff = np.load(coeff_path)

        self.chiller_capacity = float(chiller_capacity)
        self.chiller_ref_power = float(chiller_ref_power)
        self.supply_temp_ref = float(supply_temp_ref)
        self.comfort_sigma = float(comfort_sigma)

        self.chiller_high_threshold = float(chiller_high_threshold)
        self.chiller_medium_threshold = float(chiller_medium_threshold)
        self.chiller_low_threshold = float(chiller_low_threshold)

        self.f_nominal = float(f_nominal)
        self.f_cw = float(f_cw)
        self.f_tower = float(f_tower)
        self.f_chw = float(f_chw)
        self.c_p = float(c_p)
        self.density_water = float(density_water)

        self._validate_config()

    def _validate_config(self) -> None:
        if self.chiller_capacity <= 0:
            raise ValueError("chiller_capacity 必须大于 0")
        if self.chiller_ref_power <= 0:
            raise ValueError("chiller_ref_power 必须大于 0")
        if self.comfort_sigma <= 0:
            raise ValueError("comfort_sigma 必须大于 0")
        if self.f_chw <= 0:
            raise ValueError("f_chw 必须大于 0")
        if self.c_p <= 0:
            raise ValueError("c_p 必须大于 0")
        if self.density_water <= 0:
            raise ValueError("density_water 必须大于 0")

    def _get_chiller_state(self, cooling_load: float) -> list[int]:
        if cooling_load <= 0:
            return [0, 0]
        if cooling_load <= self.chiller_low_threshold:
            return [0, 1]
        if cooling_load <= self.chiller_medium_threshold:
            return [1, 0]
        if cooling_load <= self.chiller_high_threshold:
            return [1, 0]
        return [1, 1]

    def _compute_physical_terms(self, cooling_load: float, outdoor_temp: float, action_value: float) -> tuple[float, float, int, float]:
        if cooling_load <= 0:
            return 0.0, float(action_value), 0, 0.0

        on_off = self._get_chiller_state(cooling_load)
        running_chillers = max(1, int(sum(on_off)))
        clc = cooling_load / running_chillers
        t_chwr = action_value + clc / ((self.c_p * self.density_water * self.f_chw) / 3600.0)

        b1 = 4.575085e-01
        b2 = 1.313508e-01
        b3 = -4.408831e-03
        b4 = 1.930354e-02
        b5 = -5.479641e-04
        b6 = -1.376580e-03
        d1 = 6.794525e-01
        d2 = 6.694756e-02
        d3 = -3.625396e-03
        d4 = -1.018762e-02
        d5 = 1.066394e-03
        d6 = -2.113402e-03
        g1 = 7.859908e-02
        g2 = 1.950291e-01
        g3 = 7.241581e-01

        plr = float(np.clip(clc / self.chiller_capacity, 0.0, 1.5))
        t_cwr = outdoor_temp

        chiller_cap_f_temp = b1 + b2 * action_value + b3 * action_value**2 + b4 * t_cwr + b5 * t_cwr**2 + b6 * t_cwr * action_value
        chiller_eir_f_temp = d1 + d2 * action_value + d3 * action_value**2 + d4 * t_cwr + d5 * t_cwr**2 + d6 * t_cwr * action_value
        chiller_eir_f_plr = g1 + g2 * plr + g3 * plr**2

        p_chiller_single = self.chiller_ref_power * chiller_cap_f_temp * chiller_eir_f_plr * chiller_eir_f_temp
        if not np.isfinite(p_chiller_single):
            p_chiller_single = 0.0
        p_chiller_single = max(float(p_chiller_single), 0.0)
        return clc, t_chwr, running_chillers, p_chiller_single

    def compute(self, step_index: int, action_value: float) -> tuple[float, dict]:
        row = self.data.iloc[step_index]

        cooling_load = float(row["CL"]) if "CL" in row else float(row.iloc[0])
        outdoor_temp = float(row["Twb"]) if "Twb" in row else float(row.iloc[1] if len(row) > 1 else row.iloc[0])
        action_value = float(action_value)

        clc, t_chwr, chiller_count, p_chiller_single = self._compute_physical_terms(cooling_load, outdoor_temp, action_value)

        energy_score = 1.0 - p_chiller_single / self.chiller_ref_power
        comfort_score = math.exp(-0.5 * ((t_chwr - self.supply_temp_ref) / self.comfort_sigma) ** 2)

        reward = self.weight_efficiency * energy_score + self.weight_comfort * comfort_score
        info = {
            "energy_score": float(energy_score),
            "comfort_score": float(comfort_score),
            "power_chiller": float(p_chiller_single),
            "cooling_load": float(cooling_load),
            "outdoor_temp": float(outdoor_temp),
            "chiller_supply_temp": float(t_chwr),
            "chiller_count": int(chiller_count),
            "single_chiller_load": float(clc),
        }
        return float(reward), info

