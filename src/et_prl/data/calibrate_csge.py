"""南网冷站竞赛数据冷机性能模型标定（第二篇 next-paper，S0 段）。

背景：第一篇 `rewards.py` 的冷机模型是上海冷站专属标定（2 台阈值逻辑 + 上海多项式
系数），不适用于 train_A（3 台、不同容量）。本模块在 S0 校准段重新标定冷机性能模型，
产出"校准仿真器"所需的系数 + 保真度报告。

标定内容（对应 2026-10-11 S0 探查）：
1. 单台冷机功率二次多项式  p = f(PLR, t_supply, t_out)
   项: [1, plr, plr^2, t_supply, t_supply^2, t_out, t_out^2, plr*t_supply]
2. 单台容量 capacity（clc p99）
3. 台数-负荷阈值（1 台 / 2 台 / 3 台 的 CL 边界，替代上海硬编码逻辑）
4. 热平衡：回水温度 t_chwr = t_supply + clc/(cp*rho*f_chw)，标定 f_chw
5. 泵/塔功率（ΣP 组成）：非零时中位功率 + 非零比例

输出：`cleaned/calibration.json`（系数 + R²/RMSE + 阈值 + f_chw + 泵塔功率）

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


def _fit_power_model(df: pd.DataFrame) -> dict:
    """池化所有运行冷机，拟合单台功率二次多项式。"""
    rows = []
    n_on = (df[["ChAMPS01", "ChAMPS02", "ChAMPS03"]] > 0).sum(axis=1)
    cl = df["TotalRealTimeLoad"].values
    t_out = df["OutdoorWetTemp"].values
    for i in CHILLERS:
        run = df[f"ChAMPS{i}"] > 0
        if run.sum() < 100:
            continue
        d = df[run]
        clc = (d["TotalRealTimeLoad"] / n_on[run].clip(lower=1)).values
        t_supply = d[f"ChChWTempSupplySetPoint{i}"].values
        p = d[f"ChPower{i}"].values
        rows.append(np.column_stack([clc, t_supply, t_out[run.values], p]))
    if not rows:
        raise ValueError("无足够运行冷机数据")
    arr = np.vstack(rows)
    clc, t_supply, t_out, p = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # 容量估计（单台冷量 p99）
    capacity = float(np.percentile(clc, 99))
    plr = np.clip(clc / capacity, 0.0, 1.5)

    X = np.column_stack([
        np.ones(len(plr)),
        plr, plr**2,
        t_supply, t_supply**2,
        t_out, t_out**2,
        plr * t_supply,
    ])
    mask = np.isfinite(X).all(axis=1) & np.isfinite(p)
    X, p = X[mask], p[mask]
    coef, _, _, _ = np.linalg.lstsq(X, p, rcond=None)
    pred = X @ coef
    ss_res = float(((p - pred) ** 2).sum())
    ss_tot = float(((p - p.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / len(p)))
    return {
        "capacity_kw": capacity,
        "n_samples": int(len(p)),
        "power_model_terms": ["1", "plr", "plr^2", "t_supply", "t_supply^2",
                               "t_out", "t_out^2", "plr*t_supply"],
        "power_model_coeffs": [float(c) for c in coef],
        "r2": round(r2, 4),
        "rmse_kw": round(rmse, 2),
        "power_median_kw": round(float(np.median(p)), 1),
    }


def _fit_chiller_count(df: pd.DataFrame) -> dict:
    """台数-负荷阈值（从数据估计）。"""
    n_on = (df[["ChAMPS01", "ChAMPS02", "ChAMPS03"]] > 0).sum(axis=1)
    cl = df["TotalRealTimeLoad"]
    out = {}
    for k in [1, 2, 3]:
        m = n_on == k
        if m.sum() > 0:
            out[f"n{k}"] = {
                "n": int(m.sum()),
                "cl_min": round(float(cl[m].min()), 1),
                "cl_max": round(float(cl[m].max()), 1),
                "cl_median": round(float(cl[m].median()), 1),
            }
    # 阈值：1 台上限 = 1 台 CL 的 p95
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

    report = {
        "project": args.project,
        "source": str(src),
        "rows": int(len(df)),
        "power_model": _fit_power_model(df),
        "chiller_count": _fit_chiller_count(df),
        "heat_balance": _fit_heat_balance(df),
        "pump_tower": _fit_pump_tower(df),
        "physics_constants": {"cp_kj_kgk": CP, "rho_kg_m3": RHO},
    }
    out = CLEANED_DIR / "calibration.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    pm = report["power_model"]
    print(f"[calibrate] {args.project} S0: {len(df)} rows")
    print(f"  功率模型: R²={pm['r2']}  RMSE={pm['rmse_kw']} kW (中位 {pm['power_median_kw']} kW)  n={pm['n_samples']}")
    print(f"  单台容量: {pm['capacity_kw']:.0f} kW")
    print(f"  台数阈值 1->2: CL={report['chiller_count']['thr_1to2_cl_kw']} kW")
    hb = report["heat_balance"]
    if hb:
        print(f"  热平衡: ΔT中位={hb['dt_median_c']}℃  corr(冷量,ΔT)={hb['corr_clc_dt']}  f_chw={hb['f_chw_m3h']} m³/h")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
