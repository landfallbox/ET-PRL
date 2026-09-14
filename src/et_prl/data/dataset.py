"""
@Author      : landfallbox
@Date        : 2026/02/04 星期二
@Description : 数据集加载器，统一处理数据加载和预处理流程
"""

from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

from .normalizer import Normalizer
from .tensor_loader import (
    create_data_loaders,
    load_csv_to_tensor,
    reshape_to_sequence_format,
)


class DatasetLoader:
    """
    数据集加载器

    职责：
    - 加载训练、验证、测试数据
    - 执行数据预处理（reshape等）
    - 创建 DataLoader
    - 加载归一化参数
    """

    def __init__(self, config):
        """
        初始化数据集加载器

        参数：
            config: 配置对象（如 LSTMConfig）
        """
        self.config = config
        self.normalizers = None

    def load_data(
        self, load_normalizer: bool = True, reshape_for_rnn: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        加载数据并创建 DataLoader

        参数：
            load_normalizer: 是否加载归一化参数
            reshape_for_rnn: 是否将数据 reshape 为 RNN 格式 (batch, seq_len, features)

        返回：
            (train_loader, val_loader, test_loader)
        """
        # 获取数据路径
        train_path = self.config.get_train_data_path()
        val_path = self.config.get_val_data_path()
        test_path = self.config.get_test_data_path()

        # 加载张量数据
        train_features, train_targets = load_csv_to_tensor(train_path, self.config.TARGET_COLUMN)
        val_features, val_targets = load_csv_to_tensor(val_path, self.config.TARGET_COLUMN)
        test_features, test_targets = load_csv_to_tensor(test_path, self.config.TARGET_COLUMN)

        # 加载归一化参数
        if load_normalizer:
            normalizer_path = self.config.get_normalizer_path()
            self.normalizers = Normalizer.load_normalizers(normalizer_path)

        # 如果需要，reshape 为 RNN 序列格式
        if reshape_for_rnn:
            window_length = getattr(self.config, "WINDOW_LENGTH", None)
            input_size = getattr(self.config, "INPUT_SIZE", None)

            if window_length is None or input_size is None:
                raise ValueError("配置中缺少 WINDOW_LENGTH 或 INPUT_SIZE，无法 reshape 为 RNN 格式")

            train_features = reshape_to_sequence_format(train_features, window_length, input_size)
            val_features = reshape_to_sequence_format(val_features, window_length, input_size)
            test_features = reshape_to_sequence_format(test_features, window_length, input_size)

        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            train_features,
            train_targets,
            val_features,
            val_targets,
            test_features,
            test_targets,
            batch_size=self.config.BATCH_SIZE,
        )

        return train_loader, val_loader, test_loader

    def get_normalizers(self) -> Optional[Dict[str, Normalizer]]:
        """
        获取归一化器

        返回：
            归一化器字典，包含 'feature' 和 'target' 键
        """
        return self.normalizers

    def get_data_info(self) -> Dict[str, any]:
        """
        获取数据集信息

        返回：
            包含数据集路径、特征列等信息的字典
        """
        return {
            "train_path": self.config.get_train_data_path(),
            "val_path": self.config.get_val_data_path(),
            "test_path": self.config.get_test_data_path(),
            "target_column": self.config.TARGET_COLUMN,
            "batch_size": self.config.BATCH_SIZE,
        }
