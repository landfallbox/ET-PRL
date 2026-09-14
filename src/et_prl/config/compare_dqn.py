from et_prl.config.dqn import DQNConfig
from et_prl.config.gate import OnlineAnomalyDetectionConfig


class ControlCompareConfig(DQNConfig, OnlineAnomalyDetectionConfig):
    """控制策略对比评估配置，复用 DQN 与在线异常门控配置"""

    EXPERIMENT_NAME = "control_compare"
    EVAL_SUBDIR = ""
