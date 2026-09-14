from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .checkpoint_manager import CheckpointManager
from .config_manager import ConfigManager
from .logger import Logger
from .metrics_recorder import MetricsRecorder


@dataclass
class ExperimentContext:
    experiment_dir: Path
    logger: Logger
    config_manager: ConfigManager
    metrics_recorder: MetricsRecorder
    checkpoint_manager: CheckpointManager | None = None

    def close(self) -> None:
        """释放实验上下文资源（关闭 TensorBoard writer 等）。"""
        close = getattr(self.metrics_recorder, "close", None)
        if callable(close):
            close()


class ConfigLike(Protocol):
    def to_dict(self) -> dict[str, Any]:
        ...


def _resolve_config_payload(config: object) -> dict[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("config 必须提供可调用的 to_dict() 方法")

    payload = to_dict()
    if not isinstance(payload, dict):
        raise TypeError("config.to_dict() 必须返回 dict")
    return payload


def create_experiment_context(
    experiment_dir: Path,
    *,
    config: ConfigLike | object | None = None,
    config_dict: dict[str, Any] | None = None,
    save_config: bool = True,
    log_filename: str = "experiment.log",
    metrics_filename: str = "metrics.json",
    config_filename: str = "config.yaml",
    with_checkpoint_manager: bool = False,
    checkpoint_dir_name: str = "checkpoints",
) -> ExperimentContext:
    context = ExperimentContext(
        experiment_dir=Path(experiment_dir),
        logger=Logger(experiment_dir, log_filename=log_filename),
        config_manager=ConfigManager(experiment_dir, config_filename=config_filename),
        metrics_recorder=MetricsRecorder(experiment_dir, metrics_filename=metrics_filename),
        checkpoint_manager=CheckpointManager(experiment_dir, checkpoint_dir_name=checkpoint_dir_name)
        if with_checkpoint_manager
        else None,
    )

    if save_config:
        payload = config_dict
        if payload is None and config is not None:
            payload = _resolve_config_payload(config)
        if payload is not None:
            context.config_manager.save_config(payload)

    return context


def resolve_experiment_dir(
    *,
    explicit_dir: Path | None,
    experiment_name: str,
    mode: str,
    log_root_dir: Path | None = None,
) -> Path:
    if explicit_dir is not None:
        resolved = Path(explicit_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"指定的实验目录不存在: {resolved}")
        return resolved

    latest_dir = CheckpointManager.find_latest_experiment(
        experiment_name=experiment_name,
        mode=mode,
        log_root_dir=log_root_dir,
    )
    if latest_dir is None:
        raise FileNotFoundError(f"未找到实验目录: experiment={experiment_name}, mode={mode}")
    return latest_dir


def copy_config_snapshot(
    *,
    source_experiment_dir: Path,
    target_experiment_dir: Path,
    config_filename: str = "config.yaml",
    logger: Logger | None = None,
) -> bool:
    source_path = Path(source_experiment_dir) / config_filename
    target_path = Path(target_experiment_dir) / config_filename

    if not source_path.exists():
        if logger is not None:
            logger.warning(f"配置文件不存在，跳过复制: {source_path}")
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    if logger is not None:
        logger.info(f"已复制配置快照: {target_path}")
    return True
