"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : LSTM 模型评估器
"""

from typing import Any, Callable, Dict

import torch
from torch.utils.data import DataLoader

from .base import Evaluator
from .metrics import calculate_accuracy, calculate_f1


class LSTMEvaluator(Evaluator):
    """
    LSTM 模型评估器

    职责：
    - 在验证集和测试集上进行评估
    - 计算 loss 和可配置的评估指标
    """

    def __init__(self, model, loss_fn, device: str, metrics: Dict[str, Callable] = None):
        """
        初始化评估器

        参数：
            model: 深度学习模型
            loss_fn: 损失函数
            device: 计算设备
            metrics: 评估指标字典，键为指标名称，值为计算函数
                    默认使用 accuracy 和 f1
        """
        super().__init__(model, loss_fn, device)

        # 设置默认指标
        if metrics is None:
            self.metrics = {"accuracy": calculate_accuracy, "f1": calculate_f1}
        else:
            self.metrics = metrics

    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        在数据集上进行评估（用于验证和测试）

        参数：
            data_loader: 数据加载器

        返回：
            包含评估指标的字典
        """
        self.model.eval()
        total_loss = 0.0
        metric_totals = {name: 0.0 for name in self.metrics.keys()}
        batch_count = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                outputs = self.model(batch_x)
                loss = self.loss_fn(outputs, batch_y)

                # 统计 loss
                total_loss += loss.item()

                # 计算所有配置的指标
                for metric_name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, batch_y)
                    metric_totals[metric_name] += metric_value

                batch_count += 1

        # 计算平均值
        result = {"loss": total_loss / batch_count if batch_count > 0 else 0.0}

        for metric_name in self.metrics.keys():
            result[metric_name] = (
                metric_totals[metric_name] / batch_count if batch_count > 0 else 0.0
            )

        return result

    def predict(self, data_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取模型在数据集上的预测值和真实值

        参数：
            data_loader: 数据加载器

        返回：
            (predictions, targets) 元组，包含预测值和真实值张量
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                outputs = self.model(batch_x)

                # 收集预测值和真实值
                all_predictions.append(outputs.cpu())
                all_targets.append(batch_y.cpu())

        # 拼接所有批次
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        return predictions, targets
