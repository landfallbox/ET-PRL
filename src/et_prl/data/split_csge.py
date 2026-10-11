"""南网冷站竞赛数据 RSI 时间切分模块（第二篇 next-paper）。

背景：竞赛 train/test 是配置划分（test_A 与 train_A 同年重叠，test_B/C 早于 train
两三年），不符合 RSI"过去训练 → 未来部署"协议。主实验对 train_A 做三段时间切分：

- S0 校准段  2025-04-01 ~ 2025-05-31  物理模型校准 + 门控参数标定
- S1 初训段  2025-06-01 ~ 2025-06-30  训静态 π₀ 后冻结（不够则 S0+S1 合训）
- S2 部署段  2025-07-01 ~ 2025-10-20  RSI 外层环在线部署评估（不碰超参）

规则：
- 按日期切分，保留原始行（不插值跨段边界）
- 跨天缺口自然落在对应段内，按天处理时跳过
- 输出 cleaned/split/{s0,s1,s2}.csv + split_meta.json（日期范围、行数、有效天数、缺口）

运行：`uv run python -m et_prl.data.split_csge [--project A]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/external/csge_chiller_competition")
CLEANED_DIR = DATA_DIR / "cleaned"
SPLIT_DIR = CLEANED_DIR / "split"
TIME_COL = "timeStamp"

# 三段时间切分（主项目 train_A，2026-10-11 拍板）
SEGMENTS = {
    "s0": ("2025-04-01", "2025-05-31", "校准段：物理模型校准 + 门控参数标定"),
    "s1": ("2025-06-01", "2025-06-30", "初训段：训静态 π₀ 后冻结"),
    "s2": ("2025-07-01", "2025-10-20", "部署段：RSI 外层环在线部署评估"),
}


def _segment_meta(df: pd.DataFrame) -> dict:
    t = pd.to_datetime(df[TIME_COL])
    days = sorted(set(t.dt.date))
    gaps = []
    if len(days) > 1:
        prev = days[0]
        for d in days[1:]:
            if (pd.Timestamp(d) - pd.Timestamp(prev)).days > 1:
                gaps.append(f"{prev} -> {d}")
            prev = d
    return {
        "rows": int(len(df)),
        "days": int(len(days)),
        "date_min": str(days[0]) if days else None,
        "date_max": str(days[-1]) if days else None,
        "missing_days": gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="南网冷站竞赛数据 RSI 时间切分")
    parser.add_argument("--project", default="A", choices=["A", "B", "C"])
    args = parser.parse_args()

    src = CLEANED_DIR / f"train_{args.project}.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} 不存在，请先运行 clean_csge")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    meta = {"source": str(src), "project": args.project, "segments": {}}

    for name, (start, end, desc) in SEGMENTS.items():
        mask = (df[TIME_COL] >= pd.Timestamp(start)) & (df[TIME_COL] <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5))
        seg = df[mask].reset_index(drop=True)
        out = SPLIT_DIR / f"{name}.csv"
        seg.to_csv(out, index=False)
        m = _segment_meta(seg)
        m["range"] = f"{start} ~ {end}"
        m["purpose"] = desc
        meta["segments"][name] = m
        print(f"[{name}] {m['rows']} rows / {m['days']} 有效天 ({m['date_min']} ~ {m['date_max']}) 缺口 {len(m['missing_days'])} 处 -> {out}")

    meta_path = SPLIT_DIR / "split_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n元信息: {meta_path}")


if __name__ == "__main__":
    main()
