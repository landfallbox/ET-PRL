from __future__ import annotations

from pathlib import Path

from et_prl.config.control_compare import ControlCompareConfig
from et_prl.experiments.ablation.common import run_ablation_experiment


class AblationFullETPRLConfig(ControlCompareConfig):
    """Full ET-PRL: 完整模型"""

    EXPERIMENT_NAME = "ablation_full_et_prl"


def run_full_et_prl_ablation(
    train_experiment_dir: Path | None = None,
    fixed_interval: int = 1,
    gate_state_path: Path | None = None,
) -> None:
    run_ablation_experiment(
        config=AblationFullETPRLConfig,
        ablation_id="full_et_prl",
        ablation_name="Full ET-PRL",
        ablation_description="完整模型",
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
