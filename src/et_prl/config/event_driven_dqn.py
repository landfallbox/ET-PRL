from et_prl.config.dqn import DQNConfig
from et_prl.config.gate import OnlineAnomalyDetectionConfig


class EventDrivenDQNConfig(DQNConfig, OnlineAnomalyDetectionConfig):
    """事件驱动 DQN 评估配置，复用 DQN 与在线异常门控配置"""

    EXPERIMENT_NAME = "event_driven_dqn"
