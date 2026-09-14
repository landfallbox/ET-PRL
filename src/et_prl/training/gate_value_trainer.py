from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from et_prl.models import GateValueMLP


@dataclass(frozen=True)
class GateValueTrainerConfig:
    feature_columns: list[str]
    label_column: str = "y_t"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 64
    dropout: float = 0.0
    threshold: float = 0.5


class GateValueTrainer:
    """价值触发器训练器（按时间顺序切分，避免泄漏）。"""

    def __init__(self, config: GateValueTrainerConfig, device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @staticmethod
    def _build_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def _split_indices(total: int, train_ratio: float, val_ratio: float) -> tuple[slice, slice, slice]:
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        return slice(0, train_end), slice(train_end, val_end), slice(val_end, total)

    def _evaluate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_count = 0
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                probs = torch.sigmoid(logits)
                preds = (probs >= self.config.threshold).float()

                total_loss += float(loss.item()) * int(y_batch.numel())
                total_count += int(y_batch.numel())
                correct += int((preds == y_batch).sum().item())

        if total_count == 0:
            return 0.0, 0.0
        return total_loss / total_count, correct / total_count

    def train(
        self,
        dataframe: pd.DataFrame,
        model_output_path: Path,
        metadata_output_path: Path,
    ) -> tuple[Path, dict]:
        missing_columns = [
            column
            for column in self.config.feature_columns + [self.config.label_column]
            if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"训练数据缺少列: {missing_columns}")

        features = dataframe[self.config.feature_columns].to_numpy(dtype=np.float32)
        labels = dataframe[self.config.label_column].to_numpy(dtype=np.float32)

        if len(features) < 100:
            raise ValueError(f"样本过少（{len(features)}），至少需要 100 条")

        train_slice, val_slice, test_slice = self._split_indices(
            total=len(features),
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
        )
        x_train, y_train = features[train_slice], labels[train_slice]
        x_val, y_val = features[val_slice], labels[val_slice]
        x_test, y_test = features[test_slice], labels[test_slice]

        if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
            raise ValueError("训练/验证/测试划分后存在空集合，请检查比例配置")

        model = GateValueMLP(
            input_dim=x_train.shape[1],
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
        ).to(self.device)

        pos_count = float(np.sum(y_train == 1.0))
        neg_count = float(np.sum(y_train == 0.0))
        pos_weight = torch.tensor([max(1.0, neg_count / max(pos_count, 1.0))], device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        train_loader = self._build_loader(x_train, y_train, self.config.batch_size, shuffle=True)
        val_loader = self._build_loader(x_val, y_val, self.config.batch_size, shuffle=False)
        test_loader = self._build_loader(x_test, y_test, self.config.batch_size, shuffle=False)

        history: list[dict] = []
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss, train_acc = self._evaluate(model, train_loader, criterion)
            val_loss, val_acc = self._evaluate(model, val_loader, criterion)
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError("训练失败，未获得有效模型参数")

        model.load_state_dict(best_state)
        test_loss, test_acc = self._evaluate(model, test_loader, criterion)

        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": int(x_train.shape[1]),
                "hidden_size": int(self.config.hidden_size),
                "dropout": float(self.config.dropout),
                "threshold": float(self.config.threshold),
                "feature_columns": list(self.config.feature_columns),
            },
            model_output_path,
        )

        metadata = {
            "device": str(self.device),
            "train_size": int(len(x_train)),
            "val_size": int(len(x_val)),
            "test_size": int(len(x_test)),
            "positive_ratio_train": float(np.mean(y_train)),
            "best_val_loss": float(best_val_loss),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "feature_columns": list(self.config.feature_columns),
            "label_column": self.config.label_column,
            "config": {
                "train_ratio": float(self.config.train_ratio),
                "val_ratio": float(self.config.val_ratio),
                "batch_size": int(self.config.batch_size),
                "epochs": int(self.config.epochs),
                "learning_rate": float(self.config.learning_rate),
                "hidden_size": int(self.config.hidden_size),
                "dropout": float(self.config.dropout),
                "threshold": float(self.config.threshold),
            },
            "history": history,
        }
        with open(metadata_output_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        return model_output_path, metadata
