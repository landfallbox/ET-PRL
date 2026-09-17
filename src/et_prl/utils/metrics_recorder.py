"""
@Author      : landfallbox
@Date        : 2026/02/04 星期二
@Description : 指标记录器
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class MetricsRecorder:
    """
    指标记录器

    职责：
    - 记录训练和验证指标
    - 保存指标到 JSON 文件
    - 保存训练历史到 CSV 文件
    """

    def __init__(
        self,
        experiment_dir: Path,
        metrics_filename: Optional[str] = None,
        history_filename: Optional[str] = None,
        results_dir: Optional[Path] = None,
        tb_dir: Optional[Path] = None,
    ):
        """
        初始化指标记录器

        参数：
            experiment_dir: 实验目录
            metrics_filename: 指标文件名（可选，如果不提供则使用默认值）
            history_filename: 训练历史文件名（可选，如果不提供则使用默认值）
            results_dir: 训练历史 CSV 所在目录（可选，默认与 experiment_dir 相同）
            tb_dir: TensorBoard 事件文件所在目录（可选，默认与 experiment_dir 相同）
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 设置文件名，使用默认值
        if metrics_filename is None:
            metrics_filename = "metrics.json"
        if history_filename is None:
            history_filename = "training_history.csv"

        self.results_dir = Path(results_dir) if results_dir is not None else self.experiment_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = Path(tb_dir) if tb_dir is not None else self.experiment_dir

        self.metrics_filename = metrics_filename
        self.history_filename = history_filename
        self.metrics = []
        self._writer = self._init_tb_writer()

    def _init_tb_writer(self):
        """初始化 TensorBoard SummaryWriter（tensorboard 不可用时降级为 None）。"""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            return None
        return SummaryWriter(log_dir=str(self.tb_dir))

    def close(self):
        """关闭 TensorBoard writer（flush 事件文件）。"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def save_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        保存指标到 metrics.json

        参数：
            metrics: 指标字典
            step: 步骤编号（可选）
        """
        entry = {"timestamp": datetime.now().isoformat(), "step": step, **self._to_native(metrics)}

        self.metrics.append(entry)
        metrics_file = self.experiment_dir / self.metrics_filename
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

    def save_training_history(self, history: Dict[str, Any]):
        """
        保存训练历史到 CSV 文件

        参数：
            history: 训练历史字典，包含 train_loss, val_loss, train_metrics, val_metrics 等
        """
        import pandas as pd

        records = []
        num_epochs = len(history.get("train_loss", []))

        for i in range(num_epochs):
            record = {"epoch": i + 1}

            # 添加 loss
            if "train_loss" in history:
                record["train_loss"] = history["train_loss"][i]
            if "val_loss" in history:
                record["val_loss"] = history["val_loss"][i]

            # 添加其他训练指标
            if "train_metrics" in history and i < len(history["train_metrics"]):
                train_m = history["train_metrics"][i]
                for key, value in train_m.items():
                    if key != "loss":
                        record[f"train_{key}"] = value

            # 添加其他验证指标
            if "val_metrics" in history and i < len(history["val_metrics"]):
                val_m = history["val_metrics"][i]
                for key, value in val_m.items():
                    if key != "loss":
                        record[f"val_{key}"] = value

            records.append(record)

        df = pd.DataFrame(records)
        history_file = self.results_dir / self.history_filename
        df.to_csv(history_file, index=False)

        # 批量写入 TensorBoard 事件文件（训练结束时一次性生成曲线）
        self._log_records_to_tb(records)

    def _log_records_to_tb(self, records: list[Dict[str, Any]]) -> None:
        """将训练历史记录批量写入 TensorBoard（跳过 NaN，TB 不支持 NaN）。"""
        if self._writer is None or not records:
            return
        columns = list(records[0].keys())
        for i, record in enumerate(records):
            step = i + 1
            for col in columns:
                if col == "epoch":
                    continue
                value = self._to_native(record.get(col))
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if math.isnan(value) or math.isinf(value):
                        continue
                    self._writer.add_scalar(col, float(value), step)

    def _to_native(self, obj):
        """
        转换为 Python 原生类型

        参数：
            obj: 待转换对象

        返回：
            Python 原生类型对象
        """
        import numpy as np

        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_native(item) for item in obj]
        else:
            return obj
