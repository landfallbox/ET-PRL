from __future__ import annotations

from pathlib import Path

from et_prl.config.loader import load_config
from et_prl.experiments.ablation.common import run_ablation_experiment


def run_wo_dual_threshold_ablation(
    train_experiment_dir: Path | None = None,
    fixed_interval: int = 1,
    gate_state_path: Path | None = None,
) -> None:
    run_ablation_experiment(
        config=load_config("ablation/wo_dual_threshold"),
        ablation_id="wo_dual_threshold",
        ablation_name="Without Global Threshold (Local-Only)",
        ablation_description="仅保留单阈值判定，移除局部/全局融合",
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
