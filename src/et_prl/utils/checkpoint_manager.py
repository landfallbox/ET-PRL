"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 模型检查点管理器（通用可复用实现）
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """
    模型检查点管理器

    职责：
    - 保存模型、优化器状态、epoch信息等
    - 加载模型检查点
    - 管理保存路径
    """

    def __init__(self, experiment_dir: Path, checkpoint_dir_name: str = "checkpoints"):
        """
        初始化检查点管理器

        参数:
            experiment_dir: 实验目录
            checkpoint_dir_name: 检查点目录名（默认为 'checkpoints'）
        """
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir_name = checkpoint_dir_name
        self.checkpoints_dir = self.experiment_dir / checkpoint_dir_name
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]],
        filename: str = "checkpoint.pth",
        is_best: bool = False,
        best_filename: str = "best_model.pth",
    ):
        """
        保存检查点

        参数:
            model: 模型实例
            optimizer: 优化器实例
            epoch: 当前epoch
            metrics: 当前评估指标
            config: 配置字典（保存以确保可复现）
            filename: 保存的文件名
            is_best: 是否为最佳模型
            best_filename: 最佳模型的文件名
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "config": config,
        }

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        filepath = self.checkpoints_dir / filename
        torch.save(state, filepath)

        if is_best:
            best_path = self.checkpoints_dir / best_filename
            torch.save(state, best_path)

    def load_checkpoint(
        self,
        filepath: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location=None,
    ) -> Dict[str, Any]:
        """
        加载检查点到模型和优化器

        参数:
            filepath: 检查点文件路径
            model: 模型实例
            optimizer: 优化器实例（可选）
            map_location: 设备映射

        返回:
            检查点内容字典 (不包含 model_state_dict 和 optimizer_state_dict，因为已经加载)
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=map_location)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    @staticmethod
    def find_latest_experiment(
        experiment_name: str, mode: str = "train", log_root_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """
        查找指定实验的最新实验目录

        参数：
            experiment_name: 实验名称
            mode: 实验模式，"train" 或 "eval"
            log_root_dir: 日志根目录

        返回：
            最新实验目录路径，如果不存在则返回 None
        """
        if log_root_dir is None:
            log_root_dir = Path("logs")

        if mode == "train":
            subdir = "train"
        elif mode == "eval":
            subdir = "eval"
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'")

        experiment_dir = log_root_dir / experiment_name / subdir

        if not experiment_dir.exists():
            return None

        timestamp_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]

        if not timestamp_dirs:
            return None

        latest_dir = sorted(timestamp_dirs, key=lambda x: x.name)[-1]

        return latest_dir

    @staticmethod
    def load_best_model(
        experiment_dir: Path,
        model: torch.nn.Module,
        best_filename: str = "best_model.pth",
        checkpoint_dir_name: str = "checkpoints",
        map_location=None,
    ) -> Dict[str, Any]:
        """
        从实验目录加载最优模型

        参数：
            experiment_dir: 实验目录路径
            model: 模型实例
            best_filename: 最优模型文件名
            checkpoint_dir_name: 检查点目录名
            map_location: 设备映射

        返回：
            检查点信息字典
        """
        checkpoint_path = Path(experiment_dir) / checkpoint_dir_name / best_filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"最优模型文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])

        return checkpoint
