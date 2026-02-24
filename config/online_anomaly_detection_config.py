"""
@Author      : landfallbox
@Date        : 2026/02/18
@Description : 在线异常检测网关配置
"""
from config.common_config import CommonConfig


class OnlineAnomalyDetectionConfig(CommonConfig):
    """在线异常检测网关配置"""

    # ==================== 实验配置 ====================
    EXPERIMENT_NAME = "online_anomaly_detection"

    # ==================== 数据配置 ====================
    INPUT_DATA_FILE = "data/raw_data.csv"
    DATA_SUBDIR = "online_anomaly_detection"
    FEATURE_COLUMNS = ["CL", "Twb", "CL_predict"]

    # ==================== 特征工程配置 ====================
    WINDOW_SIZES = [3, 5, 12, 24]

    # ==================== 流式特征统计配置 ====================
    # EMA衰减率：越小，历史权重越大，变化越平滑
    STATS_EMA_DECAY = 0.01
    # 滑动窗口大小
    STATS_WINDOW_SIZE = 1000

    # ==================== 流式异常检测配置 ====================
    # 参考集大小（维护的样本数）
    ISOLATION_REFERENCE_SAMPLES = 500
    # 更新频率（每N个样本更新一次参考集统计）
    ISOLATION_UPDATE_FREQ = 50
    # 距离度量（euclidean 或 mahalanobis）
    ISOLATION_DISTANCE_METRIC = "euclidean"
    # 异常比例先验
    ISOLATION_CONTAMINATION = 0.2

    # ==================== 流式阈值优化配置 ====================
    # 本地阈值窗口大小（快速响应漂移）
    THRESHOLD_LOCAL_WINDOW = 100
    # 全局阈值EMA衰减率（长期稳定）
    THRESHOLD_EMA_DECAY = 0.01
    # 本地权重（0-1，越大越快响应漂移）
    # 推荐：0.7 = 70%本地 + 30%全局
    THRESHOLD_ALPHA = 0.7
    # 目标异常事件率
    TARGET_EVENT_RATE = 0.35
    # 最少样本数才执行优化
    THRESHOLD_MIN_SAMPLES = 30

    # ==================== 多尺度分布追踪配置 ====================
    # 短期窗口（快速响应短期漂移）
    DISTRIBUTION_SHORT_WINDOW = 100
    # 中期窗口（捕捉日周期）
    DISTRIBUTION_MEDIUM_WINDOW = 1440
    # 长期窗口（识别长期趋势）
    DISTRIBUTION_LONG_WINDOW = 10080
    # 分布统计EMA衰减率
    DISTRIBUTION_EMA_DECAY = 0.01

    # ==================== 异常分数融合配置 ====================
    # 短期、中期、长期分数的权重
    SCORE_SHORT_WEIGHT = 0.5  # 快速响应
    SCORE_MEDIUM_WEIGHT = 0.3  # 稳定性
    SCORE_LONG_WEIGHT = 0.2  # 趋势

    # ==================== 初始化配置 ====================
    # 是否使用离线数据进行冷启动初始化
    ENABLE_OFFLINE_INITIALIZATION = True
    # 离线初始化数据文件（可选）
    OFFLINE_INIT_DATA_FILE = "data/raw_data.csv"

    # ==================== 日志配置 ====================
    # 是否启用详细日志
    VERBOSE = True
    # 日志输出频率（每N个样本输出一次）
    LOG_FREQ = 100

