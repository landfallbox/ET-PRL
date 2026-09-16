from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from et_prl.config.loader import load_config
from et_prl.utils import create_experiment_context

from et_prl.utils.reproducibility_dqn import configure_reproducibility
from et_prl.training.dqn_trainer import DQNTrainer


def _build_run_config(base_config, run_index: int, total_runs: int):
    """基于 base 配置生成带独立 TIMESTAMP 的运行配置（frozen dataclass 用 replace）。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if total_runs > 1:
        timestamp = f"{timestamp}_run{run_index:02d}"
    return replace(base_config, TIMESTAMP=timestamp)


def train_dqn(runs: int = 1) -> None:
    if runs <= 0:
        raise ValueError(f"runs 必须为正整数，当前值: {runs}")

    for run_index in range(1, runs + 1):
        config = _build_run_config(load_config("dqn"), run_index, runs)
        experiment_dir = config.get_train_experiment_dir()
        context = create_experiment_context(
            experiment_dir=experiment_dir,
            config=config,
            log_filename=config.RUN_LOG_FILENAME,
            results_dir=config.get_run_results_dir(),
            tb_dir=config.get_run_tb_dir(),
        )

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

