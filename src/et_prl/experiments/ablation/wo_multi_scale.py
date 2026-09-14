from __future__ import annotations

from pathlib import Path

from et_prl.config.control_compare import ControlCompareConfig
from et_prl.experiments.ablation.common import run_ablation_experiment


class AblationSingleScaleShortConfig(ControlCompareConfig):
    """Without Medium/Long Scales (Short-Only)"""

    EXPERIMENT_NAME = "ablation_single_scale_short"
    GATE_SCORE_SHORT_WEIGHT = 1.0
    GATE_SCORE_MEDIUM_WEIGHT = 0.0
    GATE_SCORE_LONG_WEIGHT = 0.0


class AblationSingleScaleMediumConfig(ControlCompareConfig):
    """Without Short/Long Scales (Medium-Only)"""

    EXPERIMENT_NAME = "ablation_single_scale_medium"
    GATE_SCORE_SHORT_WEIGHT = 0.0
    GATE_SCORE_MEDIUM_WEIGHT = 1.0
    GATE_SCORE_LONG_WEIGHT = 0.0


class AblationSingleScaleLongConfig(ControlCompareConfig):
    """Without Short/Medium Scales (Long-Only)"""

    EXPERIMENT_NAME = "ablation_single_scale_long"
    GATE_SCORE_SHORT_WEIGHT = 0.0
    GATE_SCORE_MEDIUM_WEIGHT = 0.0
    GATE_SCORE_LONG_WEIGHT = 1.0


class AblationWoMultiScaleConfig(AblationSingleScaleLongConfig):
    """兼容旧命名：w/o Multi-scale（仅 long）"""

    EXPERIMENT_NAME = "ablation_wo_multi_scale"


def run_single_scale_short_ablation(
    train_experiment_dir: Path | None = None,
    fixed_interval: int = 1,
    gate_state_path: Path | None = None,
) -> None:
    run_ablation_experiment(
        config=AblationSingleScaleShortConfig,
        ablation_id="single_scale_short",
        ablation_name="Without Medium/Long Scales (Short-Only)",
        ablation_description="仅使用短尺度异常分数",
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )


def run_single_scale_medium_ablation(
    train_experiment_dir: Path | None = None,
    fixed_interval: int = 1,
    gate_state_path: Path | None = None,
) -> None:
    run_ablation_experiment(
        config=AblationSingleScaleMediumConfig,
        ablation_id="single_scale_medium",
        ablation_name="Without Short/Long Scales (Medium-Only)",
        ablation_description="仅使用中尺度异常分数",
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )


def run_single_scale_long_ablation(
    train_experiment_dir: Path | None = None,
    fixed_interval: int = 1,
    gate_state_path: Path | None = None,
) -> None:
    run_ablation_experiment(
        config=AblationSingleScaleLongConfig,
        ablation_id="single_scale_long",
        ablation_name="Without Short/Medium Scales (Long-Only)",
        ablation_description="仅使用长尺度异常分数",
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )


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
