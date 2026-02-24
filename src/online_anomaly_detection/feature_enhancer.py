"""
特征增强模块
"""
import numpy as np
from collections import deque


class OnlineFeatureEnhancer:
    """
    在线特征增强模块

    在流式处理中构建增强特征：
    - 原始特征
    - 一阶差分
    - 二阶差分
    - 多尺度滑动窗口统计（均值、标准差）
    """

    def __init__(
        self,
        feature_dim: int,
        window_sizes: list[int] | None = None,
    ):
        """
        初始化在线特征增强器

        参数：
            feature_dim: 原始特征维度
            window_sizes: 滑动窗口大小列表
        """
        if window_sizes is None:
            window_sizes = [3, 5, 12, 24]

        self.feature_dim = feature_dim
        self.window_sizes = window_sizes

        # 缓冲最近的样本用于差分和窗口统计
        max_window_size = max(window_sizes)
        self.sample_buffer = deque(maxlen=max_window_size + 1)

        # 缓冲差分
        self.diff1_buffer = deque(maxlen=max_window_size + 1)

        self.sample_count = 0

    def enhance(self, sample: np.ndarray) -> np.ndarray:
        """
        构建增强特征

        参数：
            sample: 原始特征，shape (feature_dim,)

        返回：
            增强特征，shape (enhanced_feature_dim,)
        """
        assert sample.shape[0] == self.feature_dim

        self.sample_count += 1
        self.sample_buffer.append(sample)

        features_list = [sample]

        # 一阶差分
        if len(self.sample_buffer) >= 2:
            diff1 = self.sample_buffer[-1] - self.sample_buffer[-2]
        else:
            diff1 = np.zeros(self.feature_dim)
        self.diff1_buffer.append(diff1)
        features_list.append(diff1)

        # 二阶差分
        if len(self.diff1_buffer) >= 2:
            diff2 = self.diff1_buffer[-1] - self.diff1_buffer[-2]
        else:
            diff2 = np.zeros(self.feature_dim)
        features_list.append(diff2)

        # 多尺度滑动窗口统计
        for window_size in self.window_sizes:
            if len(self.sample_buffer) >= window_size:
                window_data = np.array(list(self.sample_buffer))[-window_size:]
                rolling_mean = np.mean(window_data, axis=0)
                rolling_std = np.std(window_data, axis=0)
            else:
                rolling_mean = np.zeros(self.feature_dim)
                rolling_std = np.zeros(self.feature_dim)

            features_list.append(rolling_mean)
            features_list.append(rolling_std)

        enhanced = np.concatenate(features_list)
        return enhanced

    def get_enhanced_feature_dim(self) -> int:
        """获取增强特征维度"""
        # 原始 + 一阶差分 + 二阶差分 + (均值 + 标准差) * len(window_sizes)
        return self.feature_dim * (3 + 2 * len(self.window_sizes))

    def reset(self) -> None:
        """重置缓冲（在新段落或重启时调用）"""
        self.sample_buffer.clear()
        self.diff1_buffer.clear()
        self.sample_count = 0


