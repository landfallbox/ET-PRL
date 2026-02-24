from config.common_config import CommonConfig


class DQNConfig(CommonConfig):
    """DQN 模型配置类，继承通用配置"""

    # ==================== 实验配置 ====================
    EXPERIMENT_NAME = "dqn"

    # ==================== 数据配置 ====================
    DATA_SUBDIR = "dqn"
    STATE_COLUMNS = ["CL", "Twb", "CL_predict"]
    TRAIN_FILENAME = "train_data.csv"
    VAL_FILENAME = "val_data.csv"
    TEST_FILENAME = "test_data.csv"

    ACTION_SPACE_PATH = CommonConfig.DATA_ROOT / "action_all.npy"
    COEFF_DATE_PATH = CommonConfig.DATA_ROOT / "coeff_date.npy"
    ENV_DATA_PATH = CommonConfig.DATA_ROOT / "dqn" / "env_data_with_predict.csv"

    # ==================== 网络架构 ====================
    STATE_SIZE = 3
    ACTION_SIZE = 10
    HIDDEN_SIZES = [256]

    # ==================== 学习参数 ====================
    LEARNING_RATE = 0.001
    GAMMA = 0.95

    # ==================== 探索策略 ====================
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995

    # ==================== 经验回放 ====================
    MEMORY_CAPACITY = 50000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 100

    # ==================== 训练控制 ====================
    NUM_EPISODES = 80
    EARLY_STOPPING_PATIENCE = 5
    VAL_INTERVAL = 5

    # ==================== 奖励设计 ====================
    REWARD_WEIGHT_EFFICIENCY = 0.5
    REWARD_WEIGHT_COMFORT = 0.5
    REWARD_STRATEGY = "dataset"
    TARGET_SUPPLY_TEMP = 7.0

    # ==================== 环境参数 ====================
    CHILLER_CAPACITY = 1760
    CHILLER_REF_POWER = 314

    # ==================== 显示选项 ====================
    SHOW_FIGURES = False

    @classmethod
    def validate(cls) -> None:
        total_weight = cls.REWARD_WEIGHT_EFFICIENCY + cls.REWARD_WEIGHT_COMFORT
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Reward weights must sum to 1.0, got {total_weight}")

        if not (0 <= cls.EPSILON_MIN <= cls.EPSILON_START <= 1.0):
            raise ValueError("Epsilon values must satisfy: 0 <= EPSILON_MIN <= EPSILON_START <= 1.0")
