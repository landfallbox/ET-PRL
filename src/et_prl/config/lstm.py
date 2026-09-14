"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : LSTM 模型专用配置
"""
from et_prl.config.base import CommonConfig


class LSTMConfig(CommonConfig):
    """LSTM 模型配置类，继承通用配置"""

    # ==================== 实验配置 ====================
    EXPERIMENT_NAME = "lstm"

    # ==================== 数据配置 ====================
    # 数据特征列
    FEATURE_COLUMNS = ["CL", "Twb"]
    # 数据子目录
    DATA_SUBDIR = "lstm"
    # 时序窗口长度（滑动窗口大小）
    WINDOW_LENGTH = 24
    # 预测目标列名
    TARGET_COLUMN = "CL"

    # ==================== 模型结构配置 ====================
    # 输入特征维度
    INPUT_SIZE = 2
    # 每层 LSTM 的隐藏层神经元数量列表
    HIDDEN_SIZES = [127, 27]
    # 输出层神经元数量
    OUTPUT_SIZE = 1
    # 是否将 batch 维度放在第一维
    BATCH_FIRST = True
    # LSTM 层之间的 dropout 比例
    DROPOUT = 0.123172

    # ==================== 训练超参数配置 ====================
    LEARNING_RATE = 0.001787
    OPTIMIZER = "adam"
    LOSS_FUNCTION = "mse"
    BATCH_SIZE = 95
    EPOCHS = 77
    EARLY_STOP_PATIENCE = 10
