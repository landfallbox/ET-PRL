"""LSTM 模型训练器。"""

from torch.utils.data import DataLoader

from ..evaluation.metrics import calculate_accuracy
from .base import Trainer


class LSTMTrainer(Trainer):
    """
    LSTM 模型训练器
    """

    def __init__(self, model, optimizer, loss_fn, device: str):
        """
        初始化 LSTM 训练器

        参数：
            model: LSTM 模型
            optimizer: 优化器（如 Adam、SGD）
            loss_fn: 损失函数
            device: 计算设备（需要显式指定，通常来自 config.DEVICE）
        """
        super().__init__(model, device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """
        训练单个 epoch

        参数：
            train_loader: 训练数据加载器

        返回：
            包含训练指标的字典
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        batch_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # 前向传播
            outputs = self.model(batch_x)
            loss = self.loss_fn(outputs, batch_y)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计指标
            total_loss += loss.item()
            accuracy = calculate_accuracy(outputs, batch_y)
            total_accuracy += accuracy
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        avg_accuracy = total_accuracy / batch_count if batch_count > 0 else 0.0

        return {"loss": avg_loss, "accuracy": avg_accuracy}
