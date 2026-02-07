"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : LSTM 模型专用配置
"""
from config.common_config import CommonConfig


class LSTMConfig(CommonConfig):
    """LSTM 模型配置类，继承通用配置"""

    # ==================== 实验配置 ====================
    # 实验名称
    EXPERIMENT_NAME = "lstm"

    # ==================== 数据配置 ====================
    # 需要选择的特征列名
    FEATURE_COLUMNS = ["CL", "Twb"]  # 使用时根据实际数据设置
    # 数据划分比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    # LSTM 数据子目录
    DATA_SUBDIR = "lstm"
    # 时序窗口长度，用于构建时序特征（滑动窗口大小）
    WINDOW_LENGTH = 24
    # LSTM 预测目标列名
    TARGET_COLUMN = "CL"

    # ==================== 模型配置 ====================
    # 输入特征维度（对应 FEATURE_COLUMNS 的列数）
    INPUT_SIZE = 2
    # 每层 LSTM 的隐藏层神经元数量列表
    HIDDEN_SIZES = [127, 27]
    # 输出层神经元数量
    OUTPUT_SIZE = 1
    # 是否将 batch 维度放在第一维
    BATCH_FIRST = True
    # LSTM 层之间的 dropout 比例
    DROPOUT = 0.123172

    # ==================== LSTM 训练配置 ====================
    # 学习率
    LEARNING_RATE = 0.001787
    # 优化器类型（adam, sgd, rmsprop）
    OPTIMIZER = "adam"
    # 损失函数（mse, mae, ce）
    LOSS_FUNCTION = "mse"
    # 批大小
    BATCH_SIZE = 95
    # 训练轮数
    EPOCHS = 77
    # 早停配置：设置为正整数启用早停，设置为None禁用早停
    EARLY_STOP_PATIENCE = 10
