
from config.dqn_config import DQNConfig
from ml_toolkit.utils import ConfigManager

from src.dqn.trainer import DQNTrainer


def train_dqn() -> None:
    config = DQNConfig
    experiment_dir = config.get_train_experiment_dir()
    config_manager = ConfigManager(experiment_dir)
    config_manager.save_config(config.to_dict())

    trainer = DQNTrainer(config, experiment_dir)
    trainer.train()


if __name__ == "__main__":
    train_dqn()

