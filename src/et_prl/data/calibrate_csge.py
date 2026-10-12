"""南网冷站竞赛数据冷机性能模型标定（第二篇 next-paper，S0 段）。

背景：第一篇 `rewards.py` 的冷机模型是上海冷站专属标定（2 台阈值逻辑 + 上海多项式
系数），不适用于 train_A（3 台、不同容量）。本模块在 S0 校准段重新标定冷机性能模型，
产出"校准仿真器"所需的系数 + 保真度报告。

功率模型结构（2026-10-11 拍板，方案 D：真可分离乘积）：
    P = P_ref · f(PLR) · g(温度)
    log(P/P_ref) = a1·plr + a2·plr² + b1·ts + b2·ts² + b3·to + b4·to²
    （无 plr×ts 交互项）

为什么用可分离结构（学习 ASHRAE ARMA / 第一篇 EIR 经验）：
- 直接对功率拟合（含交互项）在温度方向产生非单调假峰（设定值近常数 + 交互项所致），
  会让 RL 策略学到"虚假最优设定"，污染外层环进化方向；
- 可分离乘积把负荷效应 f(PLR) 与温度效应 g(温度) 相乘分离，温度项单调下降
  （设定越高功率越低，物理正确），R² 几乎无损（0.862→0.859）。

标定内容：
1. 单台冷机功率可分离模型（上式）+ 单调性验证
2. 单台容量 capacity（clc p99）
3. 台数-负荷阈值（1 台 / 2 台 的 CL 边界）
4. 热平衡：回水温度 t_chwr = t_supply + clc/(cp·rho·f_chw)，标定 f_chw
5. 泵/塔功率（ΣP 组成）

输出：`cleaned/calibration.json`（系数 + R²/RMSE + 单调性 + 阈值 + f_chw + 泵塔功率）

运行：`uv run python -m et_prl.data.calibrate_csge [--project A]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLEANED_DIR = Path("data/external/csge_chiller_competition/cleaned")
SPLIT_DIR = CLEANED_DIR / "split"
TIME_COL = "timeStamp"

# 物理常数
CP = 4.186  # kJ/(kg·K) 水比热
RHO = 998.0  # kg/m3 水密度

CHILLERS = ["01", "02", "03"]


def _pool_running_chillers(df: pd.DataFrame):
    """池化所有运行冷机的 (clc, t_supply, t_out, power)。"""
    n_on = (df[["ChAMPS01", "ChAMPS02", "ChAMPS03"]] > 0).sum(axis=1)
    to_all = df["OutdoorWetTemp"].values
    rows = []
    for i in CHILLERS:
        run = df[f"ChAMPS{i}"] > 0
        if run.sum() < 100:
            continue
        d = df[run]
        clc = (d["TotalRealTimeLoad"] / n_on[run].clip(lower=1)).values
        ts = d[f"ChChWTempSupplySetPoint{i}"].values
        to = to_all[run.values]
        p = d[f"ChPower{i}"].values
        rows.append(np.column_stack([clc, ts, to, p]))
    arr = np.vstack(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _fit_power_model(clc: np.ndarray, ts: np.ndarray, to: np.ndarray, p: np.ndarray) -> dict:
    """真可分离乘积模型 + 单调性验证。"""
    capacity = float(np.percentile(clc, 99))
    plr = np.clip(clc / capacity, 0.0, 1.5)
    m = np.isfinite(np.column_stack([clc, ts, to, p])).all(axis=1) & (p > 0)
    clc, ts, to, p, plr = clc[m], ts[m], to[m], p[m], plr[m]

    p_ref = float(np.median(p))
    ylog = np.log(p / p_ref)
    # 真可分离：PLR 项 + 温度项，无交互
    X = np.column_stack([plr, plr**2, ts, ts**2, to, to**2])
    coef, _, _, _ = np.linalg.lstsq(X, ylog, rcond=None)
    pred_log = X @ coef
    pred_p = np.clip(p_ref * np.exp(pred_log), 0, None)
    r2 = 1 - ((p - pred_p) ** 2).sum() / ((p - p.mean()) ** 2).sum()
    rmse = float(np.sqrt(((p - pred_p) ** 2).mean()))

    # 单调性验证：固定 PLR=中位、t_out=中位，扫 t_supply 6→10℃
    plr_med, to_med = float(np.median(plr)), float(np.median(to))
    grid = np.arange(6.0, 10.01, 0.25)
    curve = []
    for g in grid:
        x = np.array([plr_med, plr_med**2, g, g**2, to_med, to_med**2])
        curve.append(max(float(p_ref * np.exp(x @ coef)), 0.0))
    curve = np.array(curve)
    diffs = np.diff(curve)
    n_inc = int((diffs > 0.5).sum())
    n_dec = int((diffs < -0.5).sum())
    monotonic = "单调下降" if n_inc == 0 else ("单调上升" if n_dec == 0 else "非单调")

    return {
        "structure": "separable_product",
        "formula": "P = P_ref * exp(a1*plr + a2*plr^2 + b1*ts + b2*ts^2 + b3*to + b4*to^2)",
        "capacity_kw": capacity,
        "p_ref_kw": p_ref,
        "n_samples": int(len(p)),
        "power_model_terms": ["plr", "plr^2", "t_supply", "t_supply^2", "t_out", "t_out^2"],
        "power_model_coeffs": [float(c) for c in coef],
        "r2": round(float(r2), 4),
        "rmse_kw": round(rmse, 2),
        "power_median_kw": round(float(np.median(p)), 1),
        "monotonicity": {
            "scan": "t_supply 6→10℃ @ PLR中位, t_out中位",
            "curve_kw": [round(v, 1) for v in curve[::2]],
            "verdict": monotonic,
            "n_increasing": n_inc,
            "n_decreasing": n_dec,
        },
    }


def _fit_chiller_count(df: pd.DataFrame) -> dict:
    """台数-负荷阈值（从数据估计）。"""
    n_on = (df[["ChAMPS01", "ChAMPS02", "ChAMPS03"]] > 0).sum(axis=1)
    cl = df["TotalRealTimeLoad"]
    out = {}
    for k in [1, 2, 3]:
        mm = n_on == k
        if mm.sum() > 0:
            out[f"n{k}"] = {
                "n": int(mm.sum()),
                "cl_min": round(float(cl[mm].min()), 1),
                "cl_max": round(float(cl[mm].max()), 1),
                "cl_median": round(float(cl[mm].median()), 1),
            }
    m1 = n_on == 1
    thr_1to2 = round(float(cl[m1].quantile(0.95)), 1) if m1.sum() > 0 else None
    return {"per_count": out, "thr_1to2_cl_kw": thr_1to2}


def _fit_heat_balance(df: pd.DataFrame) -> dict:
    """热平衡：标定冷冻水流量 f_chw。"""
    n_on = (df[["ChAMPS01", "ChAMPS02", "ChAMPS03"]] > 0).sum(axis=1)
    d = df[df["ChAMPS01"] > 0].copy()
    if len(d) < 100:
        return {}
    dt = (d["PriChWTempReturn01"] - d["PriChWTempSupply01"]).values
    clc = (d["TotalRealTimeLoad"] / n_on[d.index].clip(lower=1)).values
    ok = (dt > 0.3) & np.isfinite(dt) & np.isfinite(clc)
    f_est = clc[ok] * 3600.0 / (CP * RHO * dt[ok])
    return {
        "dt_median_c": round(float(np.median(dt)), 2),
        "dt_p5_c": round(float(np.percentile(dt, 5)), 2),
        "dt_p95_c": round(float(np.percentile(dt, 95)), 2),
        "corr_clc_dt": round(float(np.corrcoef(clc[ok], dt[ok])[0, 1]), 3),
        "f_chw_m3h": round(float(np.median(f_est)), 1),
    }


def _fit_pump_tower(df: pd.DataFrame) -> dict:
    """泵/塔功率（ΣP 组成）。"""
    out = {}
    for c in ["PriChWPPower01", "CWPPower01", "CTPower01"]:
        if c not in df.columns:
            continue
        s = df[c]
        nz = s[s > 0.1]
        out[c] = {
            "nonzero_frac": round(float((s > 0.1).mean()), 3),
            "median_nonzero_kw": round(float(nz.median()), 1) if len(nz) else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="南网冷站竞赛数据冷机性能模型标定")
    parser.add_argument("--project", default="A", choices=["A", "B", "C"])
    args = parser.parse_args()

    src = SPLIT_DIR / "s0.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} 不存在，请先运行 split_csge")
    df = pd.read_csv(src)

    clc, ts, to, p = _pool_running_chillers(df)
    report = {
        "project": args.project,
        "source": str(src),
        "rows": int(len(df)),
        "power_model": _fit_power_model(clc, ts, to, p),
        "chiller_count": _fit_chiller_count(df),
        "heat_balance": _fit_heat_balance(df),
        "pump_tower": _fit_pump_tower(df),
        "physics_constants": {"cp_kj_kgk": CP, "rho_kg_m3": RHO},
    }
    out = CLEANED_DIR / "calibration.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    pm = report["power_model"]
    print(f"[calibrate] {args.project} S0: {len(df)} rows")
    print(f"  功率模型(可分离): R²={pm['r2']}  RMSE={pm['rmse_kw']} kW  n={pm['n_samples']}")
    print(f"  单调性: {pm['monotonicity']['verdict']}  曲线={pm['monotonicity']['curve_kw']}")
    print(f"  单台容量: {pm['capacity_kw']:.0f} kW  P_ref={pm['p_ref_kw']:.0f} kW")
    print(f"  台数阈值 1->2: CL={report['chiller_count']['thr_1to2_cl_kw']} kW")
    hb = report["heat_balance"]
    if hb:
        print(f"  热平衡: ΔT中位={hb['dt_median_c']}℃  corr(冷量,ΔT)={hb['corr_clc_dt']}  f_chw={hb['f_chw_m3h']} m³/h")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
