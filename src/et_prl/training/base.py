"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 训练器抽象基类（通用可复用实现）
"""

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class Trainer(ABC):
    """
    训练器抽象基类

    职责：
    - 定义训练流程接口
    - 实现通用的训练循环
    - 提供模型保存/加载功能
    """

    def __init__(self, model, device: str):
        """
        初始化训练器

        参数：
            model: 神经网络模型
            device: 计算设备（需要显式指定，通常来自 config.DEVICE）
        """
        self.model = model
        self.device = device
        self.model.to(device)

    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """
        训练单个 epoch

        参数：
            train_loader: 训练数据加载器

        返回：
            包含训练指标的字典
        """
        raise NotImplementedError("Subclasses must implement this method")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        evaluator,
        epochs: int,
        logger=None,
        checkpoint_manager=None,
        config: dict = None,
        early_stop_patience: int = None,
    ) -> dict:
        """
        完整的训练流程

        参数：
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            evaluator: 评估器，用于验证
            epochs: 训练轮数
            logger: 日志器实例（可选），用于记录训练日志
            checkpoint_manager: 检查点管理器（可选），用于保存模型
            config: 配置字典（可选），用于保存到检查点
            early_stop_patience: 早停耐心值，连续多少个epoch验证loss无改善后停止训练（可选）

        返回：
            包含训练历史、最优epoch和实际停止epoch的字典
        """
        # 参数校验
        if early_stop_patience is not None:
            if not isinstance(early_stop_patience, int) or early_stop_patience <= 0:
                raise ValueError(f"early_stop_patience必须是正整数，当前值: {early_stop_patience}")

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "best_epoch": 0,
            "best_val_loss": float("inf"),
            "stopped_epoch": None,
        }

        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics.get("loss", 0))
            history["train_metrics"].append(train_metrics)

            # 验证（使用 evaluator）
            val_metrics = evaluator.validate(val_loader)
            val_loss = val_metrics.get("loss", 0)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)

            # 检查是否找到更优的模型
            if val_loss < history["best_val_loss"]:
                history["best_val_loss"] = val_loss
                history["best_epoch"] = epoch
                patience_counter = 0
                if checkpoint_manager is not None:
                    self.save_best_model(epoch, val_loss, checkpoint_manager, config, logger)
            else:
                patience_counter += 1

            # 定期记录epoch信息
            if logger is not None and (epoch + 1) % max(1, epochs // 10) == 0:
                message = (
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"train_loss: {train_metrics.get('loss', 0):.4f}, "
                    f"val_loss: {val_loss:.4f}"
                )
                logger.info(message)

            # 检查早停条件
            if early_stop_patience is not None and patience_counter >= early_stop_patience:
                history["stopped_epoch"] = epoch
                if logger is not None:
                    logger.info(f"触发早停机制: 验证loss连续 {early_stop_patience} 个epoch无改善")
                    logger.info(f"训练在第 {epoch + 1} 个epoch停止")
                break
        else:
            # 正常完成所有epoch（未触发早停）
            history["stopped_epoch"] = epochs - 1

        return history

    def save_best_model(self, epoch, val_loss, checkpoint_manager, config=None, logger=None):
        """
        保存最优模型

        参数：
            epoch: epoch索引
            val_loss: 验证loss值
            checkpoint_manager: 检查点管理器
            config: 配置字典（可选）
            logger: 日志器实例（可选）
        """
        metrics = {"val_loss": val_loss}
        checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=None,
            epoch=epoch,
            metrics=metrics,
            config=config,
            is_best=True,
        )
        if logger is not None:
            logger.info(f"保存最优模型 (Epoch {epoch + 1}, val_loss: {val_loss:.4f})")
