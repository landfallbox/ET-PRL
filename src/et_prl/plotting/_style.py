"""论文绘图统一样式。

集中定义论文图表共用的 matplotlib 基样式（字体族、mathtext、负号渲染、刻度方向），
供各 plotting 脚本复用，避免在每个脚本里重复 rcParams 字面量。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

_BASE_SANS_SERIF = ["Arial", "Helvetica", "DejaVu Sans"]


def apply_paper_style(
    unicode_minus: bool = True,
    tick_direction: str | None = None,
) -> None:
    """应用论文图表基样式到 matplotlib 全局 rcParams。

    参数：
        unicode_minus: 是否用 Unicode 负号（True 时负号渲染更美观；
            部分嵌入场景需 False 以避免字体缺字）。
        tick_direction: 刻度线方向（"in" / "out"）；None 表示不改动。
    """
    rc: dict = {
        "font.family": "sans-serif",
        "font.sans-serif": _BASE_SANS_SERIF,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": unicode_minus,
    }
    if tick_direction is not None:
        rc["xtick.direction"] = tick_direction
        rc["ytick.direction"] = tick_direction
    plt.rcParams.update(rc)
