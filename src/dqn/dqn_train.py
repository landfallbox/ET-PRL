from __future__ import annotations

from datetime import datetime

from config.dqn_config import DQNConfig
from ml_toolkit.utils import create_experiment_context

from src.dqn.reproducibility import configure_reproducibility
from src.dqn.trainer import DQNTrainer


def _build_run_config(base_config: type[DQNConfig], run_index: int, total_runs: int) -> type[DQNConfig]:
    class RunConfig(base_config):
        pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if total_runs > 1:
        timestamp = f"{timestamp}_run{run_index:02d}"
    setattr(RunConfig, "TIMESTAMP", timestamp)
    return RunConfig


def train_dqn(runs: int = 1) -> None:
    if runs <= 0:
        raise ValueError(f"runs 必须为正整数，当前值: {runs}")

    for run_index in range(1, runs + 1):
        config = _build_run_config(DQNConfig, run_index, runs)
        experiment_dir = config.get_train_experiment_dir()
        context = create_experiment_context(experiment_dir=experiment_dir, config=config)

        reproducibility = configure_reproducibility(
            seed=int(config.RANDOM_STATE),
            deterministic_cudnn=bool(config.CUDNN_DETERMINISTIC),
        )
        context.logger.info(f"训练轮次: {run_index}/{runs}")
        context.logger.info(
            "随机性配置: "
            f"seed={reproducibility['seed']}, "
            f"deterministic_cudnn={reproducibility['deterministic_cudnn']}, "
            f"cudnn_benchmark={reproducibility['cudnn_benchmark']}, "
            f"cuda_available={reproducibility['cuda_available']}"
        )

        trainer = DQNTrainer(config, experiment_dir)
        trainer.train()

