from __future__ import annotations

from pathlib import Path

from config.control_compare_config import ControlCompareConfig
from src.ablation.common import run_ablation_experiment


class AblationWoDualThresholdConfig(ControlCompareConfig):
    """w/o Dual-Threshold: 单阈值门控（移除局部/全局融合）"""

    EXPERIMENT_NAME = "ablation_wo_dual_threshold"
    GATE_ALPHA_LOCAL_WEIGHT = 1.0
    GATE_GLOBAL_EMA_DECAY = 1.0


def run_wo_dual_threshold_ablation(
    train_experiment_dir: Path | None = None,
    fixed_interval: int = 1,
    gate_state_path: Path | None = None,
) -> None:
    run_ablation_experiment(
        config=AblationWoDualThresholdConfig,
        ablation_id="wo_dual_threshold",
        ablation_name="w/o Dual-Threshold",
        ablation_description="仅保留单阈值判定，移除局部/全局融合",
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
