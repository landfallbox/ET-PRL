
from config.dqn_config import DQNConfig
from ml_toolkit.utils import create_experiment_context

from src.dqn.trainer import DQNTrainer


def train_dqn() -> None:
    config = DQNConfig
    experiment_dir = config.get_train_experiment_dir()
    create_experiment_context(experiment_dir=experiment_dir, config=config)

    trainer = DQNTrainer(config, experiment_dir)
    trainer.train()

