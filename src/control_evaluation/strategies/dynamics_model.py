"""
完整的HVAC系统环境动态模型 (Dynamics Model)

用于Model-Based Control策略，预测给定动作后的系统状态转移。
基于历史数据学习状态和动作之间的关系。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HVACDynamicsModel:
    """
    HVAC系统动态模型：预测给定当前状态和动作后，下一时刻的系统状态
    
    预测目标状态变量：
    - power_chiller: 冷机功率 (kW)
    - chiller_supply_temp: 冷机供水温度 (℃)
    - energy_score: 能效评分 (0-1)
    - comfort_score: 舒适度评分 (0-1)
    
    模型基于历史数据的统计规律和线性近似
    """

    def __init__(
        self,
        data: pd.DataFrame,
        action_space: np.ndarray,
        info_history: list[dict] | None = None,
        reward_calc: object | None = None,
        identification_sample_count: int = 256,
    ):
        """
        初始化HVAC动态模型
        
        参数：
            data: 历史状态数据（含CL, Twb, CL_predict等）
            action_space: 可用动作空间
            info_history: 历史运行数据中的info列表（用于学习模型参数）
        """
        self.data = data
        self.action_space = np.asarray(action_space, dtype=np.float32)
        self.info_history = info_history or []
        self.identification_sample_count = int(max(32, identification_sample_count))
        self._bind_objective_config(reward_calc)

        # 先设置默认参数，再用历史信息与奖励模型辨识增量校准
        self._init_heuristic_parameters()
        self._learn_model_parameters(reward_calc=reward_calc)
    
    def _learn_model_parameters(self, reward_calc: object | None = None) -> None:
        """从历史运行信息与奖励模型中学习模型参数"""
        if len(self.info_history) > 100:
            self._learn_from_info_history()
        if reward_calc is not None and len(self.data) > 0:
            self._identify_from_reward_model(reward_calc)

    def _bind_objective_config(self, reward_calc: object | None) -> None:
        """绑定目标函数参数，统一控制语义为 objective。"""
        # MBC目标改为节能优先：只优化能效，不纳入舒适度项
        self.weight_efficiency = 1.0
        self.weight_comfort = 0.0

        self.chiller_ref_power = float(getattr(reward_calc, "chiller_ref_power", 314.0))
        self.supply_temp_ref = float(getattr(reward_calc, "supply_temp_ref", 17.0))
        self.comfort_sigma = float(max(getattr(reward_calc, "comfort_sigma", 2.5), 1e-6))
        self.c_p = float(getattr(reward_calc, "c_p", 4.2))
        self.density_water = float(getattr(reward_calc, "density_water", 1000.0))
        self.f_chw = float(max(getattr(reward_calc, "f_chw", 131.0), 1e-6))
        self.low_threshold = float(getattr(reward_calc, "chiller_low_threshold", 1496.0))
        self.medium_threshold = float(getattr(reward_calc, "chiller_medium_threshold", 1760.0))
        self.high_threshold = float(getattr(reward_calc, "chiller_high_threshold", 2603.0))

    def _estimate_running_chillers(self, cooling_load: float) -> int:
        if cooling_load <= 0:
            return 0
        if cooling_load <= self.low_threshold:
            return 1
        if cooling_load <= self.medium_threshold:
            return 1
        if cooling_load <= self.high_threshold:
            return 1
        return 2
    
    def _learn_from_info_history(self) -> None:
        """从历史运行的info中估计参数，不足则回退到启发式默认值"""
        power_values = [float(info.get("power_chiller", 0.0)) 
                       for info in self.info_history if isinstance(info, dict)]
        temp_values = [float(info.get("chiller_supply_temp", 0.0)) 
                      for info in self.info_history if isinstance(info, dict)]
        
        # 使用学习的参数或启发式默认
        if power_values:
            self.power_mean = float(np.mean(power_values))
            self.power_std = float(np.std(power_values))
        else:
            self.power_mean, self.power_std = 350.0, 100.0
        
        if temp_values:
            self.temp_mean = float(np.mean(temp_values))
            self.temp_std = float(np.std(temp_values))
        else:
            self.temp_mean, self.temp_std = 7.0, 2.0

    def _identify_from_reward_model(self, reward_calc: object) -> None:
        """基于奖励模型与数据做参数辨识（单步辨识，不做多步MPC规划）"""
        if len(self.action_space) < 2 or len(self.data) == 0:
            return

        action_min = float(np.min(self.action_space))
        action_max = float(np.max(self.action_space))
        action_mid = float((action_min + action_max) / 2.0)
        probe_delta = float(max((action_max - action_min) / 10.0, 1e-3))
        a_minus = float(np.clip(action_mid - probe_delta, action_min, action_max))
        a_plus = float(np.clip(action_mid + probe_delta, action_min, action_max))
        if abs(a_plus - a_minus) < 1e-6:
            return

        sample_count = int(min(len(self.data), self.identification_sample_count))
        sample_indices = np.linspace(0, len(self.data) - 1, num=sample_count, dtype=int)

        power_sensitivity_samples: list[float] = []
        temp_sensitivity_samples: list[float] = []
        power_samples: list[float] = []
        temp_samples: list[float] = []
        cooling_load_samples: list[float] = []
        outdoor_temp_samples: list[float] = []

        for idx in sample_indices:
            try:
                _, info_minus = reward_calc.compute(int(idx), a_minus)
                _, info_plus = reward_calc.compute(int(idx), a_plus)
            except Exception:
                continue

            p_minus = float(info_minus.get("power_chiller", 0.0))
            p_plus = float(info_plus.get("power_chiller", 0.0))
            t_minus = float(info_minus.get("chiller_supply_temp", 0.0))
            t_plus = float(info_plus.get("chiller_supply_temp", 0.0))
            cl = float(info_minus.get("cooling_load", 0.0))
            twb = float(info_minus.get("outdoor_temp", 25.0))

            da = a_plus - a_minus
            if abs(da) < 1e-8:
                continue

            power_sensitivity_samples.append((p_plus - p_minus) / da)
            temp_sensitivity_samples.append((t_plus - t_minus) / da)
            power_samples.append((p_plus + p_minus) * 0.5)
            temp_samples.append((t_plus + t_minus) * 0.5)
            cooling_load_samples.append(cl)
            outdoor_temp_samples.append(twb)

        if not power_samples:
            return

        self.power_mean = float(np.mean(power_samples))
        self.power_std = float(max(np.std(power_samples), 1e-6))
        self.temp_mean = float(np.mean(temp_samples))
        self.temp_std = float(max(np.std(temp_samples), 1e-6))
        self.load_mean = float(np.mean(cooling_load_samples)) if cooling_load_samples else self.load_mean

        power_sens = float(np.median(power_sensitivity_samples)) if power_sensitivity_samples else self.power_action_sensitivity
        self.power_action_sensitivity = float(np.clip(power_sens, 1.0, 80.0))

        if temp_sensitivity_samples:
            temp_sens = float(np.median(temp_sensitivity_samples))
            if temp_sens >= 0:
                temp_sens = -abs(temp_sens)
            self.temp_action_sensitivity = float(np.clip(temp_sens, -1.5, -0.005))

        if len(cooling_load_samples) > 5:
            cl_arr = np.asarray(cooling_load_samples, dtype=np.float64)
            p_arr = np.asarray(power_samples, dtype=np.float64)
            cl_center = cl_arr - float(np.mean(cl_arr))
            p_center = p_arr - float(np.mean(p_arr))
            denom = float(np.sum(cl_center * cl_center))
            if denom > 1e-8:
                slope = float(np.sum(cl_center * p_center) / denom)
                self.power_load_coupling = float(np.clip(slope, 0.0, 1.5))

        if len(outdoor_temp_samples) > 5:
            twb_arr = np.asarray(outdoor_temp_samples, dtype=np.float64)
            t_arr = np.asarray(temp_samples, dtype=np.float64)
            twb_center = twb_arr - float(np.mean(twb_arr))
            t_center = t_arr - float(np.mean(t_arr))
            denom = float(np.sum(twb_center * twb_center))
            if denom > 1e-8:
                slope = float(np.sum(twb_center * t_center) / denom)
                self.outdoor_temp_coupling = float(np.clip(slope, -0.2, 0.2))
    
    def _init_heuristic_parameters(self) -> None:
        """使用物理启发式参数初始化模型"""
        # 功率特性
        self.power_mean = 350.0  # kW
        self.power_std = 100.0
        self.power_min = 100.0
        self.power_max = 800.0
        
        # 温度特性
        self.temp_mean = 7.0  # ℃
        self.temp_std = 2.0
        self.temp_min = 4.0
        self.temp_max = 12.0

        if "CL" in self.data.columns:
            self.load_mean = float(np.mean(self.data["CL"]))
        else:
            self.load_mean = 3500.0  # kW
        
        # 动作敏感度参数
        self._set_action_sensitivity_parameters()
    
    def _set_action_sensitivity_parameters(self) -> None:
        """设置动作敏感度参数（物理常数，不随数据变化）"""
        
        # 动作对功率的灵敏度（每个动作单位对应的功率变化）
        self.power_action_sensitivity = 25.0  # kW per action unit
        
        # 动作对温度的灵敏度（负相关：动作增加->温度下降）
        self.temp_action_sensitivity = -0.15  # ℃ per action unit
        
        # 冷负荷对功率的影响
        self.power_load_coupling = 0.3  # power_delta ~= 0.3 * load_delta

        # 室外温度对供水温度影响
        self.outdoor_temp_coupling = 0.02

        # 功率变化对供水温度影响
        self.temp_power_coupling = 0.05

        # 在线辨识增益（指数滑动）
        self.online_adapt_alpha = 0.05

        # 动作平滑惩罚系数
        self.smooth_penalty_coef = 0.005
    
    def predict_next_state(
        self,
        current_power: float,
        current_supply_temp: float,
        current_action: float,
        next_action: float,
        cooling_load: float,
        outdoor_temp: float,
    ) -> dict:
        """
        预测给定动作后的下一时刻系统状态
        
        参数：
            current_power: 当前冷机功率 (kW)
            current_supply_temp: 当前供水温度 (℃)
            current_action: 当前动作
            next_action: 下一步动作
            cooling_load: 当前冷负荷 (kW)
            outdoor_temp: 当前室外温度 (℃)
        
        返回：
            预测状态字典，包含以下键：
            - power_chiller: 预测的冷机功率
            - chiller_supply_temp: 预测的供水温度
            - energy_score: 预测的能效评分
            - comfort_score: 预测的舒适度评分
        """
        
        # 1. 预测冷机功率变化
        # 动作增加会增加冷水流量，导致功率上升
        action_delta = float(next_action - current_action)
        
        # 功率变化 = 动作影响 + 冷负荷偏离影响
        power_delta_from_action = action_delta * self.power_action_sensitivity
        power_delta_from_load = (cooling_load - self.load_mean) * self.power_load_coupling
        
        power_delta = power_delta_from_action + power_delta_from_load
        
        predicted_power = float(current_power + power_delta)
        predicted_power = float(np.clip(predicted_power, self.power_min, self.power_max))
        
        # 2. 预测供水温度变化
        outdoor_impact = (outdoor_temp - 25.0) * self.outdoor_temp_coupling
        temp_delta_from_action = action_delta * self.temp_action_sensitivity * 0.5
        temp_delta_from_power = (predicted_power - current_power) / 100.0 * self.temp_power_coupling
        temp_delta = outdoor_impact + temp_delta_from_action + temp_delta_from_power
        predicted_temp_dynamic = float(current_supply_temp + temp_delta)

        # 用冷机物理关系校正供水温度，提高动作区分度
        running_chillers = max(1, self._estimate_running_chillers(cooling_load))
        clc = float(cooling_load) / float(running_chillers)
        flow_factor = (self.c_p * self.density_water * self.f_chw) / 3600.0
        predicted_temp_physics = float(next_action + clc / max(flow_factor, 1e-6))

        predicted_temp = float(0.65 * predicted_temp_physics + 0.35 * predicted_temp_dynamic)
        predicted_temp = float(np.clip(predicted_temp, self.temp_min, self.temp_max))
        
        # 3. 预测能效评分（与评估器定义对齐）
        energy_score = float(1.0 - predicted_power / max(self.chiller_ref_power, 1e-6))
        
        # 4. 预测舒适度评分（与评估器定义对齐）
        temp_deviation = float(predicted_temp - self.supply_temp_ref)
        comfort_score = float(np.exp(-0.5 * (temp_deviation / self.comfort_sigma) ** 2))
        comfort_score = float(np.clip(comfort_score, 0.0, 1.0))
        
        return {
            "power_chiller": float(np.clip(predicted_power, 0.0, self.power_max)),
            "chiller_supply_temp": float(np.clip(predicted_temp, self.temp_min, self.temp_max)),
            "energy_score": float(np.clip(energy_score, 0.0, 1.0)),
            "comfort_score": float(np.clip(comfort_score, 0.0, 1.0)),
        }
    
    def predict_objective(
        self,
        reward_calc,
        current_step: int,
        predicted_state: dict,
        current_action: float,
        next_action: float,
    ) -> float:
        """
        基于预测状态计算控制目标值（objective）
        
        参数：
            reward_calc: RewardCalculator实例
            current_step: 当前时间步
            predicted_state: 预测的系统状态 (来自predict_next_state)
            current_action: 当前动作
            next_action: 下一步动作
        
        返回：
            预测的目标值
        """
        energy_score = float(np.clip(predicted_state.get("energy_score", 0.5), 0.0, 1.0))
        action_range = float(max(np.max(self.action_space) - np.min(self.action_space), 1e-6))
        smooth_penalty = self.smooth_penalty_coef * abs(next_action - current_action) / action_range

        predicted_objective = self.weight_efficiency * energy_score - smooth_penalty
        return float(predicted_objective)

    def update_with_observation(
        self,
        previous_action: float,
        current_action: float,
        previous_power: float,
        current_power: float,
        previous_temp: float,
        current_temp: float,
    ) -> None:
        """根据真实观测在线更新关键敏感度参数。"""
        action_delta = float(current_action - previous_action)
        if abs(action_delta) < 1e-6:
            return

        obs_power_sens = float((current_power - previous_power) / action_delta)
        obs_temp_sens = float((current_temp - previous_temp) / action_delta)

        alpha = self.online_adapt_alpha
        self.power_action_sensitivity = float(
            np.clip((1.0 - alpha) * self.power_action_sensitivity + alpha * obs_power_sens, 1.0, 80.0)
        )
        self.temp_action_sensitivity = float(
            np.clip((1.0 - alpha) * self.temp_action_sensitivity + alpha * obs_temp_sens, -1.5, -0.005)
        )
    
    def batch_predict(
        self,
        current_states: list[dict],
        action_candidates: np.ndarray,
        reward_calc,
        current_step: int,
    ) -> np.ndarray:
        """
        批量预测多个动作候选的目标值（用于动作选择）
        
        参数：
            current_states: 当前状态列表（每个元素是一个包含系统状态的dict）
            action_candidates: 候选动作数组
            reward_calc: RewardCalculator实例
            current_step: 当前时间步
        
        返回：
            预测目标数组 shape (len(action_candidates),)
        """
        if not current_states:
            return np.zeros(len(action_candidates))
        
        # 使用最新的状态
        current_state = current_states[-1]
        current_action = float(current_state.get("action", 0.0))
        current_power = float(current_state.get("power_chiller", self.power_mean))
        current_supply_temp = float(current_state.get("chiller_supply_temp", self.temp_mean))
        cooling_load = float(current_state.get("cooling_load", 0.0))
        outdoor_temp = float(current_state.get("outdoor_temp", 25.0))
        
        predicted_objectives = []
        
        for action in action_candidates:
            # 预测下一状态
            next_state = self.predict_next_state(
                current_power=current_power,
                current_supply_temp=current_supply_temp,
                current_action=current_action,
                next_action=float(action),
                cooling_load=cooling_load,
                outdoor_temp=outdoor_temp,
            )
            
            # 计算预测目标
            pred_objective = self.predict_objective(
                reward_calc=reward_calc,
                current_step=current_step,
                predicted_state=next_state,
                current_action=current_action,
                next_action=float(action),
            )
            
            predicted_objectives.append(float(pred_objective))
        
        return np.array(predicted_objectives, dtype=np.float32)
