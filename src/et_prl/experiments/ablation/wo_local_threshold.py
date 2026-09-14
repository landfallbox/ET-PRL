from __future__ import annotations

from pathlib import Path

from et_prl.config.control_compare import ControlCompareConfig
from et_prl.experiments.ablation.common import run_ablation_experiment


class AblationWoLocalThresholdConfig(ControlCompareConfig):
    """Without Local Threshold (Global-Only): 仅全局阈值门控（移除局部阈值影响）"""

    EXPERIMENT_NAME = "ablation_wo_local_threshold"
    GATE_ALPHA_LOCAL_WEIGHT = 0.0


def run_wo_local_threshold_ablation(
    train_experiment_dir: Path | None = None,
    fixed_interval: int = 1,
    gate_state_path: Path | None = None,
) -> None:
    run_ablation_experiment(
        config=AblationWoLocalThresholdConfig,
        ablation_id="wo_local_threshold",
        ablation_name="Without Local Threshold (Global-Only)",
        ablation_description="仅保留全局阈值判定，移除局部阈值影响",
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )