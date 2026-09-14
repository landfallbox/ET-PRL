"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 评估器抽象基类（通用可复用实现）
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


class Evaluator(ABC):
    """
    评估器抽象基类

    职责：
    - 在验证集上进行验证
    - 在测试集上进行测试
    - 计算和返回评估指标
    """

    def __init__(self, model, loss_fn, device: str):
        """
        初始化评估器

        参数：
            model: 深度学习模型
            loss_fn: 损失函数
            device: 计算设备（需要显式指定，通常来自 config.DEVICE）
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(device)

    @abstractmethod
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        抽象方法：在数据集上进行评估

        参数：
            data_loader: 数据加载器

        返回：
            包含评估指标的字典

        注意：
            子类实现时应使用 torch.no_grad() 上下文管理器
        """
        raise NotImplementedError("Subclasses must implement this method")

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """
        在验证集上进行验证

        参数：
            val_loader: 验证数据加载器

        返回：
            包含验证指标的字典
        """
        return self.evaluate(val_loader)

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        在测试集上进行测试

        参数：
            test_loader: 测试数据加载器

        返回：
            包含测试指标的字典
        """
        return self.evaluate(test_loader)
