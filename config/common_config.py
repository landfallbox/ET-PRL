"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 通用配置基类，供所有模型配置继承
"""
from pathlib import Path
from datetime import datetime


class CommonConfig:
    """通用配置基类，包含所有模型共用的配置"""

    # ==================== 数据配置 ====================
    # 数据根目录
    DATA_ROOT = Path("data")
    # 原始数据路径
    RAW_DATA_PATH = DATA_ROOT / "raw_data.csv"
    # 模型特定的数据子目录名（子类可覆盖）
    # 如果为空，则使用 DATA_ROOT；否则使用 DATA_ROOT / DATA_SUBDIR
    DATA_SUBDIR = ""
    # 是否打乱数据
    SHUFFLE_DATA = False
    # 随机种子
    RANDOM_STATE = 42
    # 是否启用 cuDNN 确定性（开启后可复现性更好，但可能降低性能）
    CUDNN_DETERMINISTIC = False

    # 数据划分比例（默认值，可由子类覆盖）
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # 是否使用 GPU
    USE_GPU = True
    # 设备
    DEVICE = "cuda" if USE_GPU else "cpu"

    # 时间戳（用于区分不同的实验运行）
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 日志根目录
    LOG_ROOT_DIR = Path("logs")
    # 训练实验子目录名
    TRAIN_SUBDIR = "train"
    # 评估实验子目录名
    EVAL_SUBDIR = "eval"
    # 超参优化实验子目录名
    OPTIMIZATION_SUBDIR = "optimization"

    # 训练、验证、测试数据的文件名
    TRAIN_FILENAME = "train.csv"
    VAL_FILENAME = "val.csv"
    TEST_FILENAME = "test.csv"
    # 归一化参数文件名
    NORMALIZER_FILENAME = "normalizer.json"

    # ==================== 实验输出文件配置 ====================
    # 配置文件名
    CONFIG_FILENAME = "config.yaml"
    # 实验日志文件名
    EXPERIMENT_LOG_FILENAME = "experiment.log"
    # 指标文件名
    METRICS_FILENAME = "metrics.json"
    # 训练历史文件名
    TRAINING_HISTORY_FILENAME = "training_history.csv"
    # 检查点目录名
    CHECKPOINT_DIR_NAME = "checkpoints"
    # 最佳模型文件名
    BEST_MODEL_FILENAME = "best_model.pth"
    # 最终模型文件名
    FINAL_MODEL_FILENAME = "final_model.pth"

    # ==================== 评估输出文件配置 ====================
    # 评估日志文件名
    EVALUATION_LOG_FILENAME = "evaluation.log"
    # 评估指标文件名
    EVALUATION_METRICS_FILENAME = "metrics.json"
    # 预测对比图文件名
    PREDICTION_COMPARISON_PLOT_FILENAME = "predictions_comparison.png"
    # 误差分布图文件名
    ERROR_DISTRIBUTION_PLOT_FILENAME = "error_distribution.png"
    # 预测散点图文件名
    PREDICTION_SCATTER_PLOT_FILENAME = "predictions_scatter.png"

    # ==================== 可视化配置 ====================
    # 图表输出分辨率（DPI）
    PLOT_DPI = 300
    # 预测对比图尺寸（宽, 高）
    PREDICTION_COMPARISON_FIGSIZE = (15, 6)
    # 误差分布图尺寸（宽, 高）
    ERROR_DISTRIBUTION_FIGSIZE = (10, 6)
    # 预测散点图尺寸（宽, 高）
    PREDICTION_SCATTER_FIGSIZE = (8, 8)
    # 训练历史图尺寸（宽, 高）
    TRAINING_HISTORY_FIGSIZE = (10, 6)
    # 预测对比图最大显示样本数
    MAX_PLOT_SAMPLES = 500
    # 误差分布图直方图柱数
    ERROR_HIST_BINS = 50

    @classmethod
    def get_data_dir(cls) -> Path:
        """
        获取数据目录路径

        返回：
            数据目录的完整路径
            如果 DATA_SUBDIR 为空，则返回 DATA_ROOT
            否则返回 DATA_ROOT / DATA_SUBDIR
        """
        if cls.DATA_SUBDIR:
            return cls.DATA_ROOT / cls.DATA_SUBDIR
        else:
            return cls.DATA_ROOT

    @classmethod
    def get_train_data_path(cls) -> Path:
        """获取训练数据路径"""
        return cls.get_data_dir() / cls.TRAIN_FILENAME

    @classmethod
    def get_val_data_path(cls) -> Path:
        """获取验证数据路径"""
        return cls.get_data_dir() / cls.VAL_FILENAME

    @classmethod
    def get_test_data_path(cls) -> Path:
        """获取测试数据路径"""
        return cls.get_data_dir() / cls.TEST_FILENAME

    @classmethod
    def get_normalizer_path(cls) -> Path:
        """获取归一化参数路径"""
        return cls.get_data_dir() / cls.NORMALIZER_FILENAME

    @classmethod
    def _require_experiment_name(cls) -> str:
        """
        获取并校验实验名称

        返回：
            非空实验名称
        """
        experiment_name = getattr(cls, "EXPERIMENT_NAME", "")
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("EXPERIMENT_NAME 必须在子类中设置为非空字符串")
        return experiment_name.strip()

    @classmethod
    def get_experiment_dir(cls, mode: str = "train") -> Path:
        """
        获取实验目录路径

        参数：
            mode: 实验模式，"train" 或 "eval"；为空时不添加子目录

        返回：
            实验目录的完整路径
            - 带子目录: logs/<experiment_name>/<mode>/<timestamp>/
            - 不带子目录: logs/<experiment_name>/<timestamp>/
        """
        experiment_name = cls._require_experiment_name()

        subdir = None
        if mode:
            mode_key = str(mode).lower()
            if mode_key == "train":
                subdir = cls.TRAIN_SUBDIR
            elif mode_key == "eval":
                subdir = cls.EVAL_SUBDIR
            else:
                subdir = str(mode)

        parts = [cls.LOG_ROOT_DIR, experiment_name]
        if subdir:
            parts.append(subdir)
        parts.append(cls.TIMESTAMP)

        return Path(*parts)

    @classmethod
    def get_train_experiment_dir(cls) -> Path:
        """获取训练实验目录路径"""
        return cls.get_experiment_dir(mode="train")

    @classmethod
    def get_eval_experiment_dir(cls) -> Path:
        """获取评估实验目录路径"""
        return cls.get_experiment_dir(mode="eval")

    @classmethod
    def get_optimization_dir(cls) -> Path:
        """
        获取超参优化目录路径

        返回：
            超参优化目录的完整路径，格式为 logs/<experiment_name>/optimization/

        说明：
            experiment_name 需要在子类中定义
        """
        experiment_name = cls._require_experiment_name()
        opt_dir = (
            cls.LOG_ROOT_DIR
            / experiment_name
            / cls.OPTIMIZATION_SUBDIR
        )
        return opt_dir

    @classmethod
    def to_dict(cls) -> dict:
        """
        将配置转换为字典

        返回：
            包含所有配置参数的字典
        """
        config_dict = {}
        for key in dir(cls):
            # 跳过私有属性、方法和内置属性
            if key.startswith('_'):
                continue
            value = getattr(cls, key)
            # 只保留配置值（非方法、非类）
            if not callable(value) and not isinstance(value, type):
                # 将Path对象转换为字符串，便于序列化
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict

