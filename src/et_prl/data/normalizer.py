"""数据归一化器（通用可复用实现）。"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


class Normalizer:
    """
    数据归一化器，基于Z-score标准化

    用法：
        normalizer = Normalizer()
        normalizer.fit(train_df, columns=['col1', 'col2'])
        normalized_df = normalizer.transform(df, columns=['col1', 'col2'])
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        初始化归一化器

        参数：
            epsilon: 防止除零的极小值，默认为 1e-8
        """
        self.mean: dict[str, Any] | None = None
        self.std: dict[str, Any] | None = None
        self.columns: list[str] | None = None
        self.epsilon = epsilon

    def fit(self, df: pd.DataFrame, columns: list[str]) -> None:
        """
        基于数据拟合归一化参数

        参数：
            df: 输入DataFrame（通常为训练集）
            columns: 需要归一化的列名列表
        """
        if not columns:
            raise ValueError("列名列表不能为空")

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"以下列在 DataFrame 中不存在: {missing_columns}")

        self.columns = columns
        means = df[columns].mean()
        stds = df[columns].std()
        self.mean = {col: means[col] for col in columns}
        self.std = {col: stds[col] for col in columns}

        for col in self.columns:
            if self.std[col] < self.epsilon:
                self.std[col] = self.epsilon

    def transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """
        应用归一化转换

        参数：
            df: 输入DataFrame
            columns: 需要归一化的列名列表（如果为None则使用fit时的列）

        返回：
            归一化后的DataFrame（原始DataFrame的副本）
        """
        if self.mean is None or self.std is None or self.columns is None:
            raise ValueError("Normalizer未拟合，请先调用fit()方法")

        if columns is None:
            columns = self.columns

        result_df = df.copy()
        for col in columns:
            if col not in self.mean or col not in self.std:
                raise ValueError(f"列 '{col}' 未在fit阶段处理")

            result_df[col] = (result_df[col] - self.mean[col]) / (self.std[col] + self.epsilon)

        return result_df

    @staticmethod
    def save_normalizers(normalizers: dict, path: Path) -> None:
        """
        保存多个归一化器到单个JSON文件

        参数：
            normalizers: 字典，键为归一化器名称，值为Normalizer实例
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        all_params = {}
        for name, normalizer in normalizers.items():
            if normalizer.mean is None or normalizer.std is None:
                raise ValueError(f"Normalizer '{name}' 未拟合，无法保存")

            all_params[name] = {
                "mean": normalizer.mean,
                "std": normalizer.std,
                "columns": normalizer.columns,
                "epsilon": normalizer.epsilon,
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_params, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_normalizers(path: Path) -> dict:
        """
        从单个JSON文件加载多个归一化器

        参数：
            path: 加载路径

        返回：
            字典，键为归一化器名称，值为Normalizer实例
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"归一化参数文件不存在: {path}")

        with open(path, encoding="utf-8") as f:
            all_params = json.load(f)

        normalizers = {}
        for name, params in all_params.items():
            epsilon = params.get("epsilon", 1e-8)
            normalizer = Normalizer(epsilon=epsilon)
            normalizer.mean = params["mean"]
            normalizer.std = params["std"]
            normalizer.columns = params["columns"]
            normalizers[name] = normalizer

        return normalizers
