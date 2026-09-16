"""控制对比绘图脚本共享的 results 目录解析。"""

from __future__ import annotations

from pathlib import Path


def resolve_results_dir(results_dir: Path) -> Path:
    """新布局下 step_results CSV 位于 <results_dir>/results/，旧布局直接在 <results_dir>。

    若 results_dir 本身已包含 step_results CSV（调用方直接传了 results 子目录），原样返回。
    """
    sub = Path(results_dir) / "results"
    if sub.is_dir() and any(sub.glob("*_step_results.csv")):
        return sub
    return Path(results_dir)
