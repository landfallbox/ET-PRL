"""南网冷站竞赛数据校准仿真环境（第二篇 next-paper）。

把 S0 标定的冷机性能模型扩展为"动作→反馈"闭环环境：
- 外生变量（CL/Twb/Tdb）按真实历史时序喂入
- 内生反馈（ΣP、回水温度、跟踪误差）由标定模型根据 agent 动作计算
- agent 的动作（供水温度设定）真正改变内生反馈，闭环成立

状态空间（6.3，无 LSTM）：
  [CL, Twb, Tdb, hour, day_of_week, 运行台数, PLR, 当前设定, 实际供水温度, 跟踪误差]
动作空间（6.2）：13 档聚合供水温度设定 7.0-10.0℃ × 0.25℃
奖励（6.5）：r = -w_e·(ΣP/CL) - w_c·comfort_violation - w_a·|Δsetpoint|

用法：
    env = CsgeEnv(data_df, calibration, action_space)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action_idx)

运行：`uv run python -m et_prl.environments.csge_env`（自测）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLEANED_DIR = Path("data/external/csge_chiller_competition/cleaned")
TIME_COL = "timeStamp"

# 物理常数（与 calibrate_csge 一致）
CP = 4.186
RHO = 998.0

# 动作空间：19 档 5.8-10.3℃ × 0.25℃
ACTION_MIN = 7.0
ACTION_STEP = 0.25
ACTION_N = 13
ACTION_SPACE = np.round(ACTION_MIN + np.arange(ACTION_N) * ACTION_STEP, 3)

# 状态列（按序）
STATE_COLS = [
    "CL", "Twb", "Tdb", "hour", "dow",
    "n_on", "plr", "cur_sp", "t_supply", "track_err",
]


@dataclass
class CsgeConfig:
    """环境配置（奖励权重 + 舒适带）。"""
    w_e: float = 1.0          # 能耗权重
    w_c: float = 0.5          # 舒适越界权重
    w_a: float = 0.1          # 动作变化惩罚
    comfort_lo: float = 7.5   # 舒适带下界（常数，数据 p10≈8.1 留裕量）
    comfort_hi_base: float = 9.5    # 舒适带上界（低负载，数据 p90≈9.0 留裕量）
    comfort_hi_min: float = 8.5     # 舒适带上界（高负载下限，数据 p90≈8.3 留裕量）
    comfort_hi_slope: float = 0.0025  # 上界随负载下降斜率 /kW
    comfort_hi_cl_start: float = 200.0  # 斜率起始负载
    comfort_sigma: float = 1.0

    def comfort_hi(self, cl: float) -> float:
        """负载相关舒适带上界（数据驱动：操作员负载越高设定越低）。"""
        hi = self.comfort_hi_base - self.comfort_hi_slope * max(0.0, cl - self.comfort_hi_cl_start)
        return float(max(self.comfort_hi_min, min(self.comfort_hi_base, hi)))


class CsgeEnv:
    """南网冷站校准仿真环境（回放外生 + 模型内生）。"""

    def __init__(
        self,
        data: pd.DataFrame,
        calibration: dict,
        config: CsgeConfig | None = None,
    ) -> None:
        self.cfg = config or CsgeConfig()
        self.cal = calibration
        self.data = data.reset_index(drop=True)
        self.t = pd.to_datetime(self.data[TIME_COL])
        self.n = len(self.data)

        pm = calibration["power_model"]
        self.capacity = float(pm["capacity_kw"])
        self.power_coef = np.array(pm["power_model_coeffs"])
        self.p_ref = float(pm["p_ref_kw"])
        self.track_offset = -0.1  # 实际供水温度跟踪偏移（数据中位数 act-sp≈-0.11）
        self.thr_1to2 = float(calibration["chiller_count"]["thr_1to2_cl_kw"])
        hb = calibration.get("heat_balance", {})
        self.f_chw = float(hb.get("f_chw_m3h", 87.5))
        pt = calibration.get("pump_tower", {})
        self.pump_kw = float(pt.get("PriChWPPower01", {}).get("median_nonzero_kw", 21.7))
        self.cwp_kw = float(pt.get("CWPPower01", {}).get("median_nonzero_kw", 6.9))
        self.ct_kw = float(pt.get("CTPower01", {}).get("median_nonzero_kw", 3.5))

        self._cur_sp = float(ACTION_SPACE[len(ACTION_SPACE) // 2])
        self._step = 0

    # ---------- 物理模型 ----------
    def _chiller_power(self, clc: float, t_supply: float, t_out: float) -> float:
        """单台冷机功率（标定可分离乘积模型：P = P_ref·exp(...)）。"""
        plr = float(np.clip(clc / self.capacity, 0.0, 1.5))
        x = np.array([plr, plr**2, t_supply, t_supply**2, t_out, t_out**2])
        p = self.p_ref * float(np.exp(x @ self.power_coef))
        return max(p, 0.0)

    def _n_chillers(self, cl: float) -> int:
        """台数逻辑（标定阈值）。"""
        if cl <= 0:
            return 0
        if cl <= self.thr_1to2:
            return 1
        return 2

    def _simulate(self, cl: float, t_out: float, action_sp: float) -> dict:
        """给定外生 (cl, t_out) 和动作 (供水温度设定)，算内生反馈。"""
        n = self._n_chillers(cl)
        if n == 0:
            return {"n_on": 0, "sum_p": 0.0, "t_supply": action_sp,
                    "track_err": 0.0, "clc": 0.0}
        clc = cl / n
        t_supply = action_sp + self.track_offset  # 实际供水温度≈设定+跟踪偏移（非回水温度）
        p_single = self._chiller_power(clc, action_sp, t_out)
        # ΣP = 冷机 + 泵 + 冷却塔（泵/塔按运行台数比例）
        sum_p = n * p_single + n * self.pump_kw + n * self.cwp_kw + n * self.ct_kw
        track_err = t_supply - action_sp
        return {"n_on": n, "sum_p": sum_p, "t_supply": t_supply,
                "track_err": track_err, "clc": clc}

    # ---------- 状态 ----------
    def _make_obs(self, row: pd.Series, sim: dict, ts) -> np.ndarray:
        cl = float(row["TotalRealTimeLoad"])
        twb = float(row["OutdoorWetTemp"])
        tdb = float(row["OutdoorTdbin"]) if "OutdoorTdbin" in row else twb
        hour = float(ts.hour)
        dow = float(ts.dayofweek)
        n_on = sim["n_on"]
        plr = float(np.clip(cl / (self.capacity * max(n_on, 1)), 0.0, 1.5)) if n_on > 0 else 0.0
        return np.array([
            cl, twb, tdb, hour, dow,
            float(n_on), plr, self._cur_sp, sim["t_supply"], sim["track_err"],
        ], dtype=np.float32)

    # ---------- 奖励 ----------
    def _reward(self, cl: float, sim: dict, action_sp: float) -> float:
        if cl <= 1.0:
            # 停机时段：只惩罚动作变化
            return -self.cfg.w_a * abs(action_sp - self._cur_sp)
        e = sim["sum_p"] / cl  # 单位冷量能耗 kW/kW
        # 舒适越界：供水温度出带的程度
        if sim["t_supply"] < self.cfg.comfort_lo:
            viol = (self.cfg.comfort_lo - sim["t_supply"]) / self.cfg.comfort_sigma
        elif sim["t_supply"] > self.cfg.comfort_hi(cl):
            viol = (sim["t_supply"] - self.cfg.comfort_hi(cl)) / self.cfg.comfort_sigma
        else:
            viol = 0.0
        act = abs(action_sp - self._cur_sp)
        return -(self.cfg.w_e * e + self.cfg.w_c * viol + self.cfg.w_a * act)

    # ---------- Gym 接口 ----------
    def reset(self) -> tuple[np.ndarray, dict]:
        self._step = 0
        self._cur_sp = float(ACTION_SPACE[len(ACTION_SPACE) // 2])
        row = self.data.iloc[0]
        sim = self._simulate(float(row["TotalRealTimeLoad"]),
                             float(row["OutdoorWetTemp"]), self._cur_sp)
        obs = self._make_obs(row, sim, self.t.iloc[0])
        return obs, {"step": 0}

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_sp = float(ACTION_SPACE[int(action_idx)])
        row = self.data.iloc[self._step]
        cl = float(row["TotalRealTimeLoad"])
        t_out = float(row["OutdoorWetTemp"])

        sim = self._simulate(cl, t_out, action_sp)
        reward = self._reward(cl, sim, action_sp)
        ts = self.t.iloc[self._step]

        self._cur_sp = action_sp
        self._step += 1
        done = self._step >= self.n
        obs = self._make_obs(row, sim, ts)
        info = {"sum_p": sim["sum_p"], "n_on": sim["n_on"],
                "t_supply": sim["t_supply"], "cl": cl, "step": self._step}
        return obs, float(reward), False, done, info

    @property
    def observation_size(self) -> int:
        return len(STATE_COLS)

    @property
    def action_size(self) -> int:
        return ACTION_N


def _selftest() -> None:
    cal = json.loads((CLEANED_DIR / "calibration.json").read_text())
    data = pd.read_csv(CLEANED_DIR / "split" / "s1.csv")
    env = CsgeEnv(data, cal)
    obs, _ = env.reset()
    print(f"obs shape={obs.shape}  action_size={env.action_size}")
    total_r = 0.0
    for a in range(50):
        obs, r, term, trunc, info = env.step(a % env.action_size)
        total_r += r
    print(f"50 步平均奖励={total_r / 50:.4f}  末步 info={info}")
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
