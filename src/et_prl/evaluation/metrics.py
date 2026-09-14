"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 通用评估指标函数（可复用）
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def calculate_loss(outputs: Tensor, targets: Tensor, loss_type: str = "mse") -> float:
    """
    计算损失值

    参数：
        outputs: 模型输出张量
        targets: 目标张量
        loss_type: 损失函数类型（mse, mae, ce）

    返回：
        损失值（标量）

    异常：
        ValueError: 当损失函数类型不支持时抛出
    """
    loss_type = loss_type.lower()

    if loss_type == "mse":
        loss = F.mse_loss(outputs, targets)
    elif loss_type == "mae":
        loss = F.l1_loss(outputs, targets)
    elif loss_type == "ce":
        loss = F.cross_entropy(outputs, targets)
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}，支持的类型: mse, mae, ce")

    return loss.item()


def calculate_accuracy(outputs: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    """
    计算精度（用于分类任务）

    参数：
        outputs: 模型输出张量
        targets: 目标张量
        threshold: 分类阈值，默认为 0.5

    返回：
        精度（0-1 之间的浮点数）
    """
    if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
        predictions = (outputs >= threshold).long().squeeze()
    else:
        predictions = outputs.argmax(dim=1)

    targets = targets.long().squeeze()

    correct = (predictions == targets).sum().item()
    total = targets.shape[0]

    return correct / total if total > 0 else 0.0


def calculate_f1(
    outputs: Tensor, targets: Tensor, threshold: float = 0.5, average: str = "binary"
) -> float:
    """
    计算 F1 分数（支持二分类和多分类）

    参数：
        outputs: 模型输出张量
        targets: 目标张量
        threshold: 二分类阈值，默认为 0.5
        average: 多分类平均方式，支持 'binary'（二分类）、'macro'、'micro'、'weighted'

    返回：
        F1 分数（0-1 之间的浮点数）
    """
    if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
        predictions = (outputs >= threshold).long().squeeze()
        targets = targets.long().squeeze()

        true_positive = ((predictions == 1) & (targets == 1)).sum().item()
        false_positive = ((predictions == 1) & (targets == 0)).sum().item()
        false_negative = ((predictions == 0) & (targets == 1)).sum().item()

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1
    else:
        predictions = outputs.argmax(dim=1)
        targets = targets.long().squeeze()

        if average == "binary":
            raise ValueError("多分类任务不支持 'binary' 平均方式")

        num_classes = outputs.shape[1]

        if average == "micro":
            tp = (predictions == targets).sum().item()
            total = targets.numel()
            return tp / total if total > 0 else 0.0

        f1_per_class = []
        class_weights = []

        for cls in range(num_classes):
            tp = ((predictions == cls) & (targets == cls)).sum().item()
            fp = ((predictions == cls) & (targets != cls)).sum().item()
            fn = ((predictions != cls) & (targets == cls)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_cls = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )

            f1_per_class.append(f1_cls)
            class_weights.append((targets == cls).sum().item())

        if average == "macro":
            return sum(f1_per_class) / len(f1_per_class) if f1_per_class else 0.0
        elif average == "weighted":
            total_samples = sum(class_weights)
            if total_samples == 0:
                return 0.0
            weighted_f1 = sum(f1 * weight for f1, weight in zip(f1_per_class, class_weights))
            return weighted_f1 / total_samples
        else:
            raise ValueError(f"不支持的平均方式: {average}，支持: binary, macro, micro, weighted")


def calculate_mape(outputs: Tensor, targets: Tensor) -> float:
    """
    计算平均绝对百分比误差（Mean Absolute Percentage Error）

    参数：
        outputs: 模型输出张量
        targets: 目标张量

    返回：
        MAPE 值（百分比，0-100 之间）
    """
    outputs = outputs.squeeze()
    targets = targets.squeeze()

    mask = targets != 0
    if not mask.any():
        return 0.0

    outputs = outputs[mask]
    targets = targets[mask]

    ape = torch.abs((targets - outputs) / targets)
    mape = ape.mean().item() * 100

    return mape


def calculate_precision(
    outputs: Tensor, targets: Tensor, threshold: float = 0.5, average: str = "binary"
) -> float:
    """
    计算精确率（Precision）

    参数：
        outputs: 模型输出张量
        targets: 目标张量
        threshold: 二分类阈值，默认为 0.5
        average: 多分类平均方式，支持 'binary'、'macro'、'micro'、'weighted'

    返回：
        精确率（0-1 之间的浮点数）
    """
    if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
        predictions = (outputs >= threshold).long().squeeze()
        targets = targets.long().squeeze()

        tp = ((predictions == 1) & (targets == 1)).sum().item()
        fp = ((predictions == 1) & (targets == 0)).sum().item()

        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        predictions = outputs.argmax(dim=1)
        targets = targets.long().squeeze()

        if average == "binary":
            raise ValueError("多分类任务不支持 'binary' 平均方式")

        num_classes = outputs.shape[1]

        if average == "micro":
            tp = (predictions == targets).sum().item()
            total = predictions.numel()
            return tp / total if total > 0 else 0.0

        precisions = []
        class_weights = []

        for cls in range(num_classes):
            tp = ((predictions == cls) & (targets == cls)).sum().item()
            fp = ((predictions == cls) & (targets != cls)).sum().item()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(prec)
            class_weights.append((targets == cls).sum().item())

        if average == "macro":
            return sum(precisions) / len(precisions) if precisions else 0.0
        elif average == "weighted":
            total_samples = sum(class_weights)
            if total_samples == 0:
                return 0.0
            return sum(p * w for p, w in zip(precisions, class_weights)) / total_samples
        else:
            raise ValueError(f"不支持的平均方式: {average}")


def calculate_recall(
    outputs: Tensor, targets: Tensor, threshold: float = 0.5, average: str = "binary"
) -> float:
    """
    计算召回率（Recall）

    参数：
        outputs: 模型输出张量
        targets: 目标张量
        threshold: 二分类阈值，默认为 0.5
        average: 多分类平均方式，支持 'binary'、'macro'、'micro'、'weighted'

    返回：
        召回率（0-1 之间的浮点数）
    """
    if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
        predictions = (outputs >= threshold).long().squeeze()
        targets = targets.long().squeeze()

        tp = ((predictions == 1) & (targets == 1)).sum().item()
        fn = ((predictions == 0) & (targets == 1)).sum().item()

        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        predictions = outputs.argmax(dim=1)
        targets = targets.long().squeeze()

        if average == "binary":
            raise ValueError("多分类任务不支持 'binary' 平均方式")

        num_classes = outputs.shape[1]

        if average == "micro":
            tp = (predictions == targets).sum().item()
            total = targets.numel()
            return tp / total if total > 0 else 0.0

        recalls = []
        class_weights = []

        for cls in range(num_classes):
            tp = ((predictions == cls) & (targets == cls)).sum().item()
            fn = ((predictions != cls) & (targets == cls)).sum().item()

            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(rec)
            class_weights.append((targets == cls).sum().item())

        if average == "macro":
            return sum(recalls) / len(recalls) if recalls else 0.0
        elif average == "weighted":
            total_samples = sum(class_weights)
            if total_samples == 0:
                return 0.0
            return sum(r * w for r, w in zip(recalls, class_weights)) / total_samples
        else:
            raise ValueError(f"不支持的平均方式: {average}")


def calculate_mae(outputs: Tensor, targets: Tensor) -> float:
    """
    计算平均绝对误差（Mean Absolute Error）

    参数：
        outputs: 模型输出张量
        targets: 目标张量

    返回：
        MAE 值
    """
    mae = torch.abs(outputs - targets).mean().item()
    return mae


def calculate_rmse(outputs: Tensor, targets: Tensor) -> float:
    """
    计算均方根误差（Root Mean Square Error）

    参数：
        outputs: 模型输出张量
        targets: 目标张量

    返回：
        RMSE 值
    """
    mse = ((outputs - targets) ** 2).mean().item()
    rmse = mse**0.5
    return rmse


def calculate_r2_score(outputs: Tensor, targets: Tensor) -> float:
    """
    计算 R² 分数（决定系数）

    参数：
        outputs: 模型输出张量
        targets: 目标张量

    返回：
        R² 分数
    """
    outputs = outputs.squeeze()
    targets = targets.squeeze()

    ss_res = ((targets - outputs) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()

    if ss_tot == 0:
        return 0.0

    r2 = 1 - (ss_res / ss_tot)
    return r2
