"""南网冷站竞赛数据（csge_chiller_competition）清洗模块。

用途：第二篇（next-paper）数据准备。将 `data/external/csge_chiller_competition/`
下的原始 CSV（xlsx 直转、未清洗）清洗为 `cleaned/` 子目录下的规整数据。

清洗规则（对应 TODO.md 第 1 节，2026-10-11 数据探查后细化）：
1. 哨兵值 ≈-8888.8（浮点容差 1.0）→ NaN
2. 累计电表列（`*PowerTotal*`、`CoolingCapacityTotalDaily`）→ 差分转瞬时功率；
   负差分（表计回零/跳变）→ NaN
3. 物理量级检查：温度列 [-60, 60]，频率 [0, 60]，功率 [0, 5000]，
   流量 [0, 50000]，负荷 [0, 5000]，压差 [-100, 100]；越界 → NaN
4. 时间戳解析为 datetime，重复行去重；跨天缺口（>1h）只检测记录、不填充
5. 插值：时间序列列 linear（limit 24 步 = 2h），首尾 ffill/bfill；
   非时间序列列（台数/设定值类）ffill

输出：
- `cleaned/<name>.csv`：清洗后数据
- `cleaned/cleaning_report.json`：每文件每列的处理统计

运行：`uv run python -m et_prl.data.clean_csge [--files train_A train_B train_C]`
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/external/csge_chiller_competition")
CLEANED_DIR = DATA_DIR / "cleaned"
TIME_COL = "timeStamp"
STEP = pd.Timedelta("5min")
SENTINEL_VALUE = 8888.8
SENTINEL_TOL = 1.0
MAX_INTERP_STEPS = 24  # 2 小时

# 累计电表列：差分化转瞬时值
CUMULATIVE_PATTERN = re.compile(r"PowerTotal|CoolingCapacityTotalDaily")

# 物理量级边界（按列名前缀匹配，单位：℃ / Hz / kW / m³/h / kW(负荷) / kPa）
PHYSICAL_RANGES: list[tuple[re.Pattern, float, float]] = [
    (re.compile(r"Temp|WetTemp|Tdbin|Evap|Cond"), -60.0, 60.0),
    (re.compile(r"Freq"), 0.0, 60.0),
    (re.compile(r"Power$|^ChAMPS"), 0.0, 5000.0),
    (re.compile(r"Flow"), 0.0, 50000.0),
    (re.compile(r"RealTimeLoad|CoolingCapacity$"), 0.0, 5000.0),
    (re.compile(r"DiffPress|DP$|Press"), -1000.0, 1000.0),
]

# 非时间序列列（台数、设定值）：只做 ffill，不做 linear 插值
STEPWISE_PATTERN = re.compile(r"SetPoint|Setting|num$")


@dataclass
class ColumnReport:
    """单列清洗统计。"""

    sentinel: int = 0
    out_of_range: int = 0
    cumulative_diff_neg: int = 0
    filled: int = 0
    remaining_nan: int = 0


@dataclass
class FileReport:
    """单文件清洗统计。"""

    name: str
    rows_raw: int = 0
    rows_out: int = 0
    time_dups_removed: int = 0
    time_gaps: list[str] = field(default_factory=list)  # "起->止" 的跨天缺口描述
    rows_dropped_gap: int = 0
    columns: dict[str, ColumnReport] = field(default_factory=dict)


def _match_sentinel(s: pd.Series) -> pd.Series:
    return (s.abs() - SENTINEL_VALUE).abs() < SENTINEL_TOL


def _physical_range(col: str) -> tuple[float, float] | None:
    if col == "OutdoorWetTemp":
        return -30.0, 60.0  # 湿球温度物理下限，滤掉 -50 类坏值
    for pattern, lo, hi in PHYSICAL_RANGES:
        if pattern.search(col):
            return lo, hi
    return None


def _detect_gaps(df: pd.DataFrame) -> list[str]:
    """检测跨天时间缺口（>1h 视为缺口），返回描述列表。不修改数据。"""
    gaps = []
    diffs = df.index.to_series().diff()
    for ts, d in diffs.items():
        if d > pd.Timedelta("1h"):
            prev = ts - d
            gaps.append(f"{prev:%Y-%m-%d %H:%M} -> {ts:%Y-%m-%d %H:%M} ({d / pd.Timedelta('1D'):.1f}d)")
    return gaps


def clean_file(path: Path, out_dir: Path) -> FileReport:
    """清洗单个 CSV 文件，写出 cleaned/<name>.csv，返回统计报告。"""
    report = FileReport(name=path.stem)
    df = pd.read_csv(path)
    report.rows_raw = len(df)

    # 1) 时间戳
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    df = df.set_index(TIME_COL).sort_index()
    report.time_dups_removed = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")]

    # 2) 检测跨天缺口（不填充，保留原始时间点）
    report.time_gaps = _detect_gaps(df)

    for col in df.columns:
        s = df[col]
        cr = report.columns.setdefault(col, ColumnReport())

        # 3) 哨兵值
        mask_sent = _match_sentinel(s)
        cr.sentinel = int(mask_sent.sum())
        s = s.mask(mask_sent)

        # 4) 累计列差分
        if CUMULATIVE_PATTERN.search(col):
            diff = s.diff()
            cr.cumulative_diff_neg = int((diff < 0).sum())
            s = diff.mask(diff < 0)
            s.iloc[0] = np.nan  # 差分首值无意义
            df[col] = s
            s = df[col]

        # 5) 物理量级检查（累计列差分化后按瞬时功率检查，滤表计跳变）
        if CUMULATIVE_PATTERN.search(col):
            rng = (0.0, 5000.0)
        else:
            rng = _physical_range(col)
        if rng is not None:
            lo, hi = rng
            mask_oor = (s < lo) | (s > hi)
            cr.out_of_range = int(mask_oor.sum())
            s = s.mask(mask_oor)
            df[col] = s
            s = df[col]

        # 6) 插值 / 填充
        n_before = int(s.isna().sum())
        if STEPWISE_PATTERN.search(col):
            s = s.ffill().bfill()
        else:
            s = s.interpolate(method="linear", limit=MAX_INTERP_STEPS, limit_area="inside")
            s = s.ffill().bfill()
        df[col] = s

        cr.filled = n_before - int(s.isna().sum())
        cr.remaining_nan = int(s.isna().sum())

    report.rows_out = len(df)
    out_path = out_dir / f"{path.stem}.csv"
    df.to_csv(out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗南网冷站竞赛数据")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["train_A", "train_B", "train_C", "test_A", "test_B", "test_C"],
        help="要清洗的文件名（不含 .csv）",
    )
    args = parser.parse_args()

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.files:
        src = DATA_DIR / f"{name}.csv"
        if not src.exists():
            print(f"[skip] {src} 不存在")
            continue
        print(f"[clean] {name} ...")
        rep = clean_file(src, CLEANED_DIR)
        reports.append(rep)
        total_sent = sum(c.sentinel for c in rep.columns.values())
        total_oor = sum(c.out_of_range for c in rep.columns.values())
        total_neg = sum(c.cumulative_diff_neg for c in rep.columns.values())
        total_nan = sum(c.remaining_nan for c in rep.columns.values())
        print(
            f"  rows {rep.rows_raw} -> {rep.rows_out} (去重 {rep.time_dups_removed}, "
            f"跨天缺口 {len(rep.time_gaps)}) | 哨兵 {total_sent} 越界 {total_oor} "
            f"表计回零 {total_neg} 残余NaN {total_nan}"
        )
        for g in rep.time_gaps:
            print(f"    [gap] {g}")
        for col, c in rep.columns.items():
            if c.remaining_nan > 0:
                print(f"    [warn] {col}: 残余 NaN {c.remaining_nan}")

    report_path = CLEANED_DIR / "cleaning_report.json"
    report_path.write_text(
        json.dumps(
            [
                {
                    "name": r.name,
                    "rows_raw": r.rows_raw,
                    "rows_out": r.rows_out,
                    "time_gaps": r.time_gaps,
                    "time_dups_removed": r.time_dups_removed,
                    "columns": {k: vars(v) for k, v in r.columns.items()},
                }
                for r in reports
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
