"""门控超参优化的搜索空间与阶段管理。

本模块只含搜索空间构建、阶段参数定义、动态范围计算、前阶段最优加载、
trial 配置生成等构建逻辑，不含优化主流程。供 gate_hyperparams_optimize.py 使用。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from et_prl.config.control_compare import ControlCompareConfig
from et_prl.utils import HyperparameterSpace, Logger

# 门控优化参数 -> (配置属性名, 类型) 的单一事实源。
# 搜索空间构建、trial 配置覆盖、多阶段参数合并均从此派生，
# 新增/调整门控超参时只需在此处维护一次。
GATE_OPT_PARAM_MAPPING: dict[str, tuple[str, type]] = {
    "global_ema_decay": ("GATE_GLOBAL_EMA_DECAY", float),
    "alpha_local_weight": ("GATE_ALPHA_LOCAL_WEIGHT", float),
    "local_window_size": ("GATE_LOCAL_WINDOW_SIZE", int),
    "reference_samples": ("GATE_REFERENCE_SAMPLES", int),
    "threshold_bias": ("GATE_THRESHOLD_BIAS", float),
    "threshold_quantile": ("THRESHOLD_QUANTILE", float),
    "threshold_mad_scale": ("THRESHOLD_MAD_SCALE", float),
    "threshold_local_update_rate": ("THRESHOLD_LOCAL_UPDATE_RATE", float),
    "threshold_quantile_weight": ("THRESHOLD_QUANTILE_WEIGHT", float),
    "threshold_min_samples_for_optimization": ("THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION", int),
    "score_short_weight": ("GATE_SCORE_SHORT_WEIGHT", float),
    "score_medium_weight": ("GATE_SCORE_MEDIUM_WEIGHT", float),
    "trigger_hysteresis_margin": ("GATE_TRIGGER_HYSTERESIS_MARGIN", float),
}


def _get_dynamic_param_range(
    param_name: str,
    config: ControlCompareConfig,
    phase: str,
    previous_best_value: float | None,
    shrink_ratio: float = 0.2,
) -> tuple[float | int, float | int]:
    """
    根据阶段和前阶段最优值，计算动态搜索范围

    参数：
        param_name: 参数名（如'global_ema_decay'）
        config: 配置类
        phase: 当前阶段（'phase1', 'phase2', 'phase3'）
        previous_best_value: 前阶段的最优值（None表示phase1或不需要动态缩小）
        shrink_ratio: 范围缩小比例（默认±20%）
    """
    min_attr = f"GATE_OPT_{param_name.upper()}_MIN"
    max_attr = f"GATE_OPT_{param_name.upper()}_MAX"

    default_min = getattr(config, min_attr)
    default_max = getattr(config, max_attr)

    # Phase 1: 使用原始范围
    if phase == "phase1" or previous_best_value is None:
        return default_min, default_max

    # Phase 2/3: 在前阶段最优值周围缩小范围
    range_width = default_max - default_min
    shrink_amount = range_width * shrink_ratio

    new_min = max(default_min, previous_best_value - shrink_amount)
    new_max = min(default_max, previous_best_value + shrink_amount)

    return new_min, new_max


def _get_phase_params(phase: str) -> dict:
    """获取每个阶段应该优化的参数列表"""
    params_config = {
        "phase1": {
            "description": "Layer A: 核心决策参数（5个）",
            "params": [
                "threshold_bias",
                "trigger_hysteresis_margin",
                "threshold_quantile",
                "threshold_mad_scale",
                "threshold_local_update_rate",
            ],
        },
        "phase2": {
            "description": "Layer B: 自适应参数（5个）",
            "params": [
                "alpha_local_weight",
                "global_ema_decay",
                "threshold_min_samples_for_optimization",
                "local_window_size",
                "reference_samples",
            ],
        },
        "phase3": {
            "description": "Layer C+D: 融合与风格参数（3个）",
            "params": [
                "score_short_weight",
                "score_medium_weight",
                "threshold_quantile_weight",
            ],
        },
    }
    return params_config.get(phase, params_config["phase1"])


def _load_previous_phase_best(
    output_dir: Path,
    current_phase: str,
    previous_phase_result_dir: Path | None = None,
    logger: Logger | None = None,
) -> dict | None:
    """从前一个阶段加载最优参数

    Args:
        output_dir: 当前phase的输出目录
        current_phase: 当前阶段（'phase1', 'phase2', 'phase3'）
        previous_phase_result_dir: 显式指定的前阶段结果目录路径
        logger: 日志对象，用于详细的输出信息

    Returns:
        前阶段最优参数dict，或None（phase1或找不到结果）
    """
    if current_phase == "phase1":
        if logger:
            logger.info("[Phase 1] 初始阶段，无前阶段结果可继承")
        return None

    phase_order = {"phase2": "phase1", "phase3": "phase2"}
    prev_phase = phase_order.get(current_phase)
    if prev_phase is None:
        return None

    # 确定前阶段最优参数文件的路径
    candidate_paths: list[Path] = []
    if previous_phase_result_dir is not None:
        # 显式路径兼容两种输入：
        # 1) .../phase1
        # 2) .../phase1/gate
        candidate_paths = [
            previous_phase_result_dir / "results" / "best_params.json",
            previous_phase_result_dir / "best_params.json",
            previous_phase_result_dir / "gate" / "results" / "best_params.json",
            previous_phase_result_dir / "gate" / "best_params.json",
        ]
        source_hint = f"(显式指定: {previous_phase_result_dir})"
    else:
        # 回退：自动检测（相同timestamp）下的标准gate目录
        candidate_paths = [
            output_dir.parent / prev_phase / "gate" / "results" / "best_params.json",
            output_dir.parent / prev_phase / "gate" / "best_params.json",
            output_dir.parent / prev_phase / "results" / "best_params.json",
            output_dir.parent / prev_phase / "best_params.json",
        ]
        source_hint = f"(自动检测: {output_dir.parent})"

    prev_best_path = next((path for path in candidate_paths if path.exists()), None)

    if prev_best_path is None:
        if logger:
            logger.warning(
                f"[{current_phase.upper()}] 前阶段结果不存在 {source_hint}，"
                f"尝试路径: {[str(p) for p in candidate_paths]}"
            )
            logger.warning(
                f"[{current_phase.upper()}] 这可能意味着 {prev_phase} 尚未运行，请先执行该阶段"
            )
        return None

    try:
        with open(prev_best_path) as f:
            params = json.load(f)
        if logger:
            logger.info(
                f"[{current_phase.upper()}] 已加载 {prev_phase.upper()} 的最优参数 {source_hint}"
            )
            logger.info(f"[{current_phase.upper()}] 文件: {prev_best_path}")
            logger.info(f"[{current_phase.upper()}] 参数: {list(params.keys())}")
        return params
    except Exception as exc:
        if logger:
            logger.error(
                f"[{current_phase.upper()}] 加载前阶段最优参数失败: {prev_best_path} | 异常: {exc}"
            )
        return None


def _create_search_space(
    config: ControlCompareConfig,
    phase: str = "phase1",
    previous_best_params: dict | None = None,
    shrink_ratio: float = 0.2,
) -> HyperparameterSpace:
    """
    构建超参搜索空间（支持多阶段）

    参数：
        config: 配置类
        phase: 当前优化阶段 ('phase1', 'phase2', 'phase3')
        previous_best_params: 前一阶段的最优参数
        shrink_ratio: 搜索范围缩小比例（相对于原始范围）
    """
    phase_cfg = _get_phase_params(phase)
    params_to_optimize = phase_cfg["params"]

    space = HyperparameterSpace()

    # 按顺序添加参数（类型取自 GATE_OPT_PARAM_MAPPING 单一事实源）
    for param_name in params_to_optimize:
        if param_name not in GATE_OPT_PARAM_MAPPING:
            continue

        param_type = GATE_OPT_PARAM_MAPPING[param_name][1]

        # 如果不是phase1，尝试从previous_best_params获取最优值并缩小范围
        prev_value = None
        if previous_best_params is not None and param_name in previous_best_params:
            prev_value = previous_best_params[param_name]

        min_val, max_val = _get_dynamic_param_range(
            param_name,
            config,
            phase,
            prev_value,
            shrink_ratio,
        )

        if param_type == "float":
            space.add_float(param_name, min_val, max_val)
        else:  # int
            space.add_int(param_name, int(min_val), int(max_val))

    return space


def _build_trial_config(base_config, params: dict):
    """基于 base 配置实例生成 trial 配置（frozen dataclass 用 replace）。"""
    # 参数映射取自 GATE_OPT_PARAM_MAPPING 单一事实源
    param_config_mapping = GATE_OPT_PARAM_MAPPING

    # 针对score_weight需要特别处理
    has_score_short = "score_short_weight" in params
    has_score_medium = "score_medium_weight" in params

    updates: dict = {}
    for param_name, (config_attr, param_type) in param_config_mapping.items():
        if param_name in params:
            # 使用优化进来的值
            value = param_type(params[param_name])
        else:
            # 使用base_config的默认值
            value = param_type(getattr(base_config, config_attr))
        updates[config_attr] = value

    # 计算score_long_weight
    if has_score_short and has_score_medium:
        score_short_weight = float(params["score_short_weight"])
        score_medium_weight = float(params["score_medium_weight"])
    else:
        score_short_weight = float(base_config.GATE_SCORE_SHORT_WEIGHT)
        score_medium_weight = float(base_config.GATE_SCORE_MEDIUM_WEIGHT)

    score_long_weight = max(0.01, 1.0 - score_short_weight - score_medium_weight)
    updates["GATE_SCORE_LONG_WEIGHT"] = float(score_long_weight)

    return replace(base_config, **updates)
