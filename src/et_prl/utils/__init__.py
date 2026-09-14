"""通用工具库。"""

from .checkpoint_manager import CheckpointManager
from .config_manager import ConfigManager
from .logger import Logger
from .loss_factory import create_loss_fn
from .metrics_recorder import MetricsRecorder
from .optimizer_factory import create_optimizer
from .orchestration import (
    ExperimentContext,
    copy_config_snapshot,
    create_experiment_context,
    resolve_experiment_dir,
)
from .reproducibility import configure_reproducibility
from .rate_controller import RateController

__all__ = [
    "Logger",
    "CheckpointManager",
    "ConfigManager",
    "MetricsRecorder",
    "create_loss_fn",
    "create_optimizer",
    "configure_reproducibility",
    "ExperimentContext",
    "create_experiment_context",
    "resolve_experiment_dir",
    "copy_config_snapshot",
    "RateController",
]

# 可选的可视化和超参数优化工具（需要额外依赖）
try:
    from .visualizer import Visualizer

    __all__.append("Visualizer")
except ImportError:
    pass

try:
    from .hyperparameter_optimizer import BayesianOptimizer, HyperparameterSpace

    __all__.extend(["HyperparameterSpace", "BayesianOptimizer"])
except ImportError:
    pass
