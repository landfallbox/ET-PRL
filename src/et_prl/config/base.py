"""
配置基类（frozen dataclass schema）。

设计要点（见 MIGRATION_PLAN.md 2.4）：
- YAML 是单一事实来源；dataclass 仅作 schema 校验，字段本身不承载默认值。
- 字段名与旧类属性名一致（大写），键与 YAML 一一对应。
- 路径字段（DATA_ROOT / LOG_ROOT_DIR）为 Path，loader 将 YAML 相对路径解析为
  项目根绝对路径；to_dict() 再序列化为相对路径，保证快照与输入 YAML 一致。
- 派生值（DEVICE / RAW_DATA_PATH / 各实验目录）改为 property / 实例方法。
- TIMESTAMP 为字段：由 loader 注入当前时间戳，运行期用 dataclasses.replace 覆盖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path


def project_root() -> Path:
    """项目根目录。优先取环境变量 ET_PRL_ROOT，否则按本文件位置推导。"""
    env = os.environ.get("ET_PRL_ROOT")
    if env:
        return Path(env).resolve()
    # src/et_prl/config/base.py -> parents[3] == 项目根
    return Path(__file__).resolve().parents[3]


def _to_relative_str(value: Path, root: Path) -> str:
    """将 Path 序列化为相对项目根的字符串（无法相对化时退回绝对路径）。"""
    try:
        return value.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


@dataclass(frozen=True)
class BaseConfig:
    """通用配置基类，包含所有模型共用的配置（schema）。"""

    # ==================== 数据配置 ====================
    DATA_ROOT: Path
    DATA_SUBDIR: str
    SHUFFLE_DATA: bool
    RANDOM_STATE: int
    CUDNN_DETERMINISTIC: bool
    TRAIN_RATIO: float
    VAL_RATIO: float
    TEST_RATIO: float

    # ==================== 温度舒适范围 ====================
    COMFORT_LOWER_BOUND: float
    COMFORT_UPPER_BOUND: float

    # ==================== 设备与运行 ====================
    USE_GPU: bool
    TIMESTAMP: str

    # ==================== 实验目录 ====================
    LOG_ROOT_DIR: Path
    TRAIN_SUBDIR: str
    EVAL_SUBDIR: str
    OPTIMIZATION_SUBDIR: str
    EXPERIMENT_NAME: str

    # ==================== 数据文件名 ====================
    TRAIN_FILENAME: str
    VAL_FILENAME: str
    TEST_FILENAME: str
    NORMALIZER_FILENAME: str

    # ==================== 实验输出文件名 ====================
    CONFIG_FILENAME: str
    RUN_LOG_FILENAME: str
    METRICS_FILENAME: str
    TRAINING_HISTORY_FILENAME: str
    CHECKPOINT_DIR_NAME: str
    BEST_MODEL_FILENAME: str
    FINAL_MODEL_FILENAME: str
    RESULTS_DIR_NAME: str
    FIGURES_DIR_NAME: str
    TB_DIR_NAME: str

    # ==================== 评估输出文件名 ====================
    PREDICTION_COMPARISON_PLOT_FILENAME: str
    ERROR_DISTRIBUTION_PLOT_FILENAME: str
    PREDICTION_SCATTER_PLOT_FILENAME: str

    # ==================== 可视化配置 ====================
    PLOT_DPI: int
    PREDICTION_COMPARISON_FIGSIZE: tuple
    ERROR_DISTRIBUTION_FIGSIZE: tuple
    PREDICTION_SCATTER_FIGSIZE: tuple
    TRAINING_HISTORY_FIGSIZE: tuple
    MAX_PLOT_SAMPLES: int
    ERROR_HIST_BINS: int

    # ==================== 派生值（property） ====================
    @property
    def root(self) -> Path:
        """项目根目录（绝对路径）。"""
        return project_root()

    @property
    def DEVICE(self) -> str:
        """计算设备：USE_GPU 为真且 CUDA 可用时返回 cuda，否则 cpu（fail-safe）。"""
        if not self.USE_GPU:
            return "cpu"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def RAW_DATA_PATH(self) -> Path:
        """原始数据路径，由 DATA_ROOT 派生。"""
        return self.DATA_ROOT / "raw_data.csv"

    # ==================== 运行目录子目录（方法） ====================
    def get_run_results_dir(self, mode: str = "train") -> Path:
        """运行目录下的结果数据子目录（step_results、训练历史、报告等）。"""
        return self.get_experiment_dir(mode) / self.RESULTS_DIR_NAME

    def get_run_figures_dir(self, mode: str = "train") -> Path:
        """运行目录下的图表子目录。"""
        return self.get_experiment_dir(mode) / self.FIGURES_DIR_NAME

    def get_run_tb_dir(self, mode: str = "train") -> Path:
        """运行目录下的 TensorBoard 事件子目录。"""
        return self.get_experiment_dir(mode) / self.TB_DIR_NAME

    # ==================== 路径方法 ====================
    def get_data_dir(self) -> Path:
        """数据目录：DATA_ROOT 或 DATA_ROOT/DATA_SUBDIR。"""
        if self.DATA_SUBDIR:
            return self.DATA_ROOT / self.DATA_SUBDIR
        return self.DATA_ROOT

    def get_train_data_path(self) -> Path:
        return self.get_data_dir() / self.TRAIN_FILENAME

    def get_val_data_path(self) -> Path:
        return self.get_data_dir() / self.VAL_FILENAME

    def get_test_data_path(self) -> Path:
        return self.get_data_dir() / self.TEST_FILENAME

    def get_normalizer_path(self) -> Path:
        return self.get_data_dir() / self.NORMALIZER_FILENAME

    def get_experiment_dir(self, mode: str = "train") -> Path:
        """实验目录：LOG_ROOT_DIR/EXPERIMENT_NAME/[subdir]/TIMESTAMP。"""
        if not self.EXPERIMENT_NAME.strip():
            raise ValueError("EXPERIMENT_NAME 必须为非空字符串")
        subdir = None
        if mode:
            mode_key = str(mode).lower()
            if mode_key == "train":
                subdir = self.TRAIN_SUBDIR
            elif mode_key == "eval":
                subdir = self.EVAL_SUBDIR
            else:
                subdir = str(mode)
        parts = [self.LOG_ROOT_DIR, self.EXPERIMENT_NAME]
        if subdir:
            parts.append(subdir)
        parts.append(self.TIMESTAMP)
        return Path(*parts)

    def get_train_experiment_dir(self) -> Path:
        return self.get_experiment_dir(mode="train")

    def get_eval_experiment_dir(self) -> Path:
        return self.get_experiment_dir(mode="eval")

    def get_optimization_dir(self) -> Path:
        return self.LOG_ROOT_DIR / self.EXPERIMENT_NAME / self.OPTIMIZATION_SUBDIR

    # ==================== 校验与序列化 ====================
    def validate(self) -> None:
        """配置一致性校验钩子，子类可覆盖。基类默认无约束。"""
        return None

    def to_dict(self) -> dict:
        """导出全量生效字段为可序列化 dict（Path 转相对项目根字符串）。"""
        root = self.root
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                out[f.name] = _to_relative_str(value, root)
            else:
                out[f.name] = value
        return out
