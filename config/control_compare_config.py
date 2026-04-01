from config.dqn_config import DQNConfig
from config.online_anomaly_detection_config import OnlineAnomalyDetectionConfig


class ControlCompareConfig(DQNConfig, OnlineAnomalyDetectionConfig):
    """控制策略对比评估配置，复用 DQN 与在线异常门控配置"""

    EXPERIMENT_NAME = "control_compare"
    EVAL_SUBDIR = ""
