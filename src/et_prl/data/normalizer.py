"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 数据归一化器（通用可复用实现）
"""

import json
from pathlib import Path

import pandas as pd


class Normalizer:
    """
    数据归一化器，基于Z-score标准化

    用法：
        normalizer = Normalizer()
        normalizer.fit(train_df, columns=['col1', 'col2'])
        normalized_df = normalizer.transform(df, columns=['col1', 'col2'])
        normalizer.save(save_path)

        normalizer2 = Normalizer()
        normalizer2.load(save_path)
        normalized_df2 = normalizer2.transform(df, columns=['col1', 'col2'])
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        初始化归一化器

        参数：
            epsilon: 防止除零的极小值，默认为 1e-8
        """
        self.mean = None
        self.std = None
        self.columns = None
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
        self.mean = df[columns].mean().to_dict()
        self.std = df[columns].std().to_dict()

        for col in self.columns:
            if self.std[col] < self.epsilon:
                self.std[col] = self.epsilon

    def transform(self, df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
        """
        应用归一化转换

        参数：
            df: 输入DataFrame
            columns: 需要归一化的列名列表（如果为None则使用fit时的列）

        返回：
            归一化后的DataFrame（原始DataFrame的副本）
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer未拟合，请先调用fit()方法")

        if columns is None:
            columns = self.columns

        result_df = df.copy()
        for col in columns:
            if col not in self.mean or col not in self.std:
                raise ValueError(f"列 '{col}' 未在fit阶段处理")

            result_df[col] = (result_df[col] - self.mean[col]) / (self.std[col] + self.epsilon)

        return result_df

    def inverse_transform(self, df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
        """
        反归一化（恢复原始数据）

        参数：
            df: 已归一化的DataFrame
            columns: 需要反归一化的列名列表（如果为None则使用fit时的列）

        返回：
            反归一化后的DataFrame（原始DataFrame的副本）
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer未拟合，请先调用fit()方法")

        if columns is None:
            columns = self.columns

        result_df = df.copy()
        for col in columns:
            if col not in self.mean or col not in self.std:
                raise ValueError(f"列 '{col}' 未在fit阶段处理")

            result_df[col] = result_df[col] * (self.std[col] + self.epsilon) + self.mean[col]

        return result_df

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        拟合并应用归一化转换（fit + transform的便捷方法）

        参数：
            df: 输入DataFrame
            columns: 需要归一化的列名列表

        返回：
            归一化后的DataFrame（原始DataFrame的副本）
        """
        self.fit(df, columns)
        return self.transform(df, columns)

    def __repr__(self) -> str:
        """返回归一化器的字符串表示"""
        if self.mean is None or self.std is None:
            return f"Normalizer(fitted=False, epsilon={self.epsilon})"
        return f"Normalizer(fitted=True, columns={self.columns}, epsilon={self.epsilon})"

    def save(self, path: Path) -> None:
        """
        保存归一化参数到JSON文件

        参数：
            path: 保存路径
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer未拟合，无法保存")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            "mean": self.mean,
            "std": self.std,
            "columns": self.columns,
            "epsilon": self.epsilon,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

    def load(self, path: Path) -> None:
        """
        从JSON文件加载归一化参数

        参数：
            path: 加载路径
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"归一化参数文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.mean = params["mean"]
        self.std = params["std"]
        self.columns = params["columns"]
        self.epsilon = params.get("epsilon", 1e-8)

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

        with open(path, "r", encoding="utf-8") as f:
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
