
from config.dqn_config import DQNConfig
from ml_toolkit.utils import configure_reproducibility, create_experiment_context

from src.dqn.trainer import DQNTrainer


def train_dqn() -> None:
    config = DQNConfig
    experiment_dir = config.get_train_experiment_dir()
    context = create_experiment_context(experiment_dir=experiment_dir, config=config)

    reproducibility = configure_reproducibility(
        seed=int(config.RANDOM_STATE),
        deterministic_cudnn=bool(config.CUDNN_DETERMINISTIC),
    )
    context.logger.info(
        "随机性配置: "
        f"seed={reproducibility['seed']}, "
        f"deterministic_cudnn={reproducibility['deterministic_cudnn']}, "
        f"cudnn_benchmark={reproducibility['cudnn_benchmark']}, "
        f"cuda_available={reproducibility['cuda_available']}"
    )

    trainer = DQNTrainer(config, experiment_dir)
    trainer.train()

