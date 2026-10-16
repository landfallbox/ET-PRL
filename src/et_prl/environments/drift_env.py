"""合成漂移测试床：在标定物理模型上注入"设备退化"潜漂移。

功率模型 P = p_ref·exp(a1·plr + a2·plr² + b1·ts + b2·ts² + b3·tout + b4·tout²)
的最优供水温度 ts* = -b1/(2b2)。设备退化使 b1 变负、b2 变大，ts* 随时间下移。
漂移是潜变量：观测（CL/Twb/...）不变，只有奖励（功率）暴露它——
策略必须从功率反馈里感知并适应，这正是 RSI 的目标场景。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from et_prl.environments.csge_env import CsgeEnv


def lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * float(np.clip(f, 0.0, 1.0))


class DriftEnv(CsgeEnv):
    """带时间漂移的标定仿真环境。

    drift: None（= 普通 CsgeEnv）或 dict(b1_0, b1_1, b2_0, b2_1)。
    t0_frac / t_span: 本段数据在全局漂移轴上的起点/跨度（用于周窗口切片）。
    """

    def __init__(self, data, calibration, config=None,
                 drift: dict | None = None, t0_frac: float = 0.0, t_span: float = 1.0):
        super().__init__(data, calibration, config)
        self._drift = drift
        self._t0 = t0_frac
        self._tspan = t_span

    def _drift_frac(self) -> float:
        return self._t0 + (self._step / max(self.n, 1)) * self._tspan

    def _chiller_power(self, clc: float, t_supply: float, t_out: float) -> float:
        if self._drift is None:
            return super()._chiller_power(clc, t_supply, t_out)
        f = self._drift_frac()
        coef = self.power_coef.copy()
        coef[2] = lerp(self._drift["b1_0"], self._drift["b1_1"], f)
        coef[3] = lerp(self._drift["b2_0"], self._drift["b2_1"], f)
        plr = float(np.clip(clc / self.capacity, 0.0, 1.5))
        x = np.array([plr, plr**2, t_supply, t_supply**2, t_out, t_out**2])
        return max(self.p_ref * float(np.exp(x @ coef)), 0.0)

    def ts_star(self) -> float:
        """当前步的最优供水温度（无约束）。"""
        if self._drift is None:
            b1, b2 = self.power_coef[2], self.power_coef[3]
        else:
            f = self._drift_frac()
            b1 = lerp(self._drift["b1_0"], self._drift["b1_1"], f)
            b2 = lerp(self._drift["b2_0"], self._drift["b2_1"], f)
        return -b1 / (2 * b2) if b2 != 0 else float("inf")
