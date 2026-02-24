from config.common_config import CommonConfig


class OnlineAnomalyDetectionConfig(CommonConfig):
    """在线异常检测与门控配置"""

    EXPERIMENT_NAME = "online_anomaly_detection"

    # 输入特征
    FEATURE_COLUMNS = ["CL", "Twb", "CL_predict"]

    # OnlineFeatureEnhancer
    FEATURE_WINDOW_SIZES = [3, 5, 12, 24]

    # StreamingAnomalyGate
    GATE_LOCAL_WINDOW_SIZE = 100
    GATE_GLOBAL_EMA_DECAY = 0.01
    GATE_REFERENCE_SAMPLES = 500
    GATE_CONTAMINATION = 0.2
    GATE_ALPHA_LOCAL_WEIGHT = 0.7

    # StreamingStats
    STATS_EMA_DECAY = 0.01
    STATS_WINDOW_SIZE = 1000

    # StreamingIsolationDepth
    ISOLATION_UPDATE_FREQ = 50
    ISOLATION_DISTANCE_METRIC = "euclidean"
    ISOLATION_DECAY_STRATEGY = "FIFO"

    # StreamingThresholdOptimizer
    THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION = 30

    # MultiScaleDistributionTracker
    TRACKER_WINDOWS = {
        "short": 100,
        "medium": 1440,
        "long": 10080,
    }
    TRACKER_EMA_DECAY = 0.01
