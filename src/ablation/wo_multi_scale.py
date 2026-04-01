from __future__ import annotations

from pathlib import Path

from config.control_compare_config import ControlCompareConfig
from src.ablation.common import run_ablation_experiment


class AblationWoMultiScaleConfig(ControlCompareConfig):
    """w/o Multi-scale: 单尺度异常分数（仅 long）"""

    EXPERIMENT_NAME = "ablation_wo_multi_scale"
    GATE_SCORE_SHORT_WEIGHT = 0.0
    GATE_SCORE_MEDIUM_WEIGHT = 0.0
    GATE_SCORE_LONG_WEIGHT = 1.0


def run_wo_multi_scale_ablation(
    train_experiment_dir: Path | None = None,
    fixed_interval: int = 1,
    gate_state_path: Path | None = None,
) -> None:
    run_ablation_experiment(
        config=AblationWoMultiScaleConfig,
        ablation_id="wo_multi_scale",
        ablation_name="w/o Multi-scale",
        ablation_description="移除多尺度异常融合，使用单尺度异常分数",
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
