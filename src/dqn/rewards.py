from __future__ import annotations

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
    ) -> None:
        self.data = data
        self.action_min = float(np.min(action_space))
        self.action_max = float(np.max(action_space))
        self.action_range = max(self.action_max - self.action_min, 1e-6)
        self.weight_efficiency = weight_efficiency
        self.weight_comfort = weight_comfort
        self.target_supply_temp = target_supply_temp

        self.has_reward = "reward" in data.columns
        self.has_energy = "energy_score" in data.columns
        self.has_comfort = "comfort_score" in data.columns

        if "CL" in data.columns:
            cl_range = float(data["CL"].max() - data["CL"].min())
        else:
            cl_range = 0.0
        if "CL_predict" in data.columns:
            pred_range = float(data["CL_predict"].max() - data["CL_predict"].min())
        else:
            pred_range = 0.0

        self.cl_range = max(cl_range, 1e-6)
        self.pred_range = max(pred_range, 1e-6)

    def compute(self, step_index: int, action_value: float) -> tuple[float, dict]:
        row = self.data.iloc[step_index]

        if self.has_reward:
            reward = float(row["reward"])
            energy_score = float(row["energy_score"]) if self.has_energy else 0.0
            comfort_score = float(row["comfort_score"]) if self.has_comfort else 0.0
            info = {
                "energy_score": energy_score,
                "comfort_score": comfort_score,
            }
            return reward, info

        if self.has_energy:
            energy_score = float(row["energy_score"])
        else:
            energy_score = 1.0 - min(abs(action_value - self.target_supply_temp) / self.action_range, 1.0)

        if self.has_comfort:
            comfort_score = float(row["comfort_score"])
        elif "CL" in row and "CL_predict" in row:
            diff = abs(float(row["CL_predict"]) - float(row["CL"]))
            comfort_score = 1.0 - min(diff / self.pred_range, 1.0)
        else:
            comfort_score = 0.0

        reward = self.weight_efficiency * energy_score + self.weight_comfort * comfort_score
        info = {
            "energy_score": energy_score,
            "comfort_score": comfort_score,
        }
        return float(reward), info

