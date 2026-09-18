from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import yaml
from et_prl.utils import CheckpointManager, ConfigManager, Logger, resolve_experiment_dir

from et_prl.config.dqn import DQNConfig
from et_prl.config.loader import get_default
from et_prl.data import load_action_space, load_state_data
from et_prl.agents.dqn import DQNAgent
from et_prl.agents.dqn.rewards import RewardCalculator
from et_prl.detection.streaming_gate import StreamingAnomalyGate


TestDataSplit = Literal["test", "val", "train"]


def resolve_train_experiment_dir(train_experiment_dir: Path | None) -> Path:
    dqn_config = get_default("dqn")
    return resolve_experiment_dir(
        explicit_dir=train_experiment_dir,
        experiment_name=dqn_config.EXPERIMENT_NAME,
        mode="train",
        log_root_dir=dqn_config.LOG_ROOT_DIR,
    )


def copy_train_config(
    resolved_train_dir: Path,
    test_experiment_dir: Path,
    config: DQNConfig,
    logger: Logger,
) -> None:
    train_config_manager = ConfigManager(resolved_train_dir, config_filename=config.CONFIG_FILENAME)
    test_config_manager = ConfigManager(test_experiment_dir, config_filename=config.CONFIG_FILENAME)

    try:
        train_config = train_config_manager.load_config()
    except FileNotFoundError:
        train_config = {}
        logger.warning(f"训练配置文件不存在，使用当前配置类快照: {resolved_train_dir / config.CONFIG_FILENAME}")
    except Exception as err:
        # YAML parser may raise ConstructorError or other subclasses of YAMLError.
        if isinstance(err, yaml.YAMLError):
            config_path = resolved_train_dir / config.CONFIG_FILENAME
            logger.warning(f"解析训练配置时出错，尝试使用 FullLoader: {err}")
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    train_config = yaml.load(f, Loader=yaml.FullLoader) or {}
                logger.info("已使用 FullLoader 成功加载训练配置")
            except Exception as ex:
                logger.warning(f"使用 FullLoader 仍然失败，忽略训练配置: {ex}")
                train_config = {}
        else:
            # re-raise unexpected errors
            raise

    # 优先保留训练快照中的参数（如已调优的 DQN 参数），并补齐 compare 流程涉及模块的缺失超参（如门控参数）。
    merged_config = {
        **config.to_dict(),
        **train_config,
    }
    test_config_manager.save_config(merged_config)
    logger.info(f"已写入合并配置快照: {test_experiment_dir / config.CONFIG_FILENAME}")


def _create_agent(config: DQNConfig, action_space: np.ndarray, device: torch.device) -> DQNAgent:
    return DQNAgent.from_config(config=config, action_space=action_space, device=device)


def build_test_components(
    config: DQNConfig,
    resolved_train_dir: Path,
    data_split: TestDataSplit = "test",
) -> tuple[pd.DataFrame, np.ndarray, DQNAgent, dict, RewardCalculator]:
    split_key = str(data_split).lower().strip()
    if split_key == "test":
        split_data_path = config.get_test_data_path()
    elif split_key == "val":
        split_data_path = config.get_val_data_path()
    elif split_key == "train":
        split_data_path = config.get_train_data_path()
    else:
        raise ValueError(f"不支持的数据划分: {data_split}")

    test_data = load_state_data(split_data_path, config.STATE_COLUMNS)
    if test_data.empty:
        raise ValueError(f"{split_key} 数据为空: {split_data_path}")

    action_space = load_action_space(config.ACTION_SPACE_PATH)
    device = torch.device(config.DEVICE)

    agent = _create_agent(config, action_space, device)
    checkpoint = CheckpointManager.load_best_model(
        experiment_dir=resolved_train_dir,
        model=agent.policy_net,
        best_filename=config.BEST_MODEL_FILENAME,
        checkpoint_dir_name=config.CHECKPOINT_DIR_NAME,
        map_location=device,
    )
    agent.policy_net.eval()

    reward_calc = RewardCalculator.from_config(config=config, data=test_data, action_space=action_space)

    return test_data, action_space, agent, checkpoint, reward_calc


def create_streaming_gate(
    config: Any,
    test_data: pd.DataFrame,
    logger: Logger,
    gate_state_path: Path | None,
) -> StreamingAnomalyGate:
    feature_columns = config.FEATURE_COLUMNS
    missing_columns = [column for column in feature_columns if column not in test_data.columns]
    if missing_columns:
        raise ValueError(f"测试数据缺少异常门控特征列: {missing_columns}")

    score_long_weight = getattr(config, "GATE_SCORE_LONG_WEIGHT", None)
    if score_long_weight is None:
        score_long_weight = max(0.01, 1.0 - float(config.GATE_SCORE_SHORT_WEIGHT) - float(config.GATE_SCORE_MEDIUM_WEIGHT))

    gate = StreamingAnomalyGate(
        feature_dim=len(feature_columns),
        local_window_size=config.GATE_LOCAL_WINDOW_SIZE,
        global_ema_decay=config.GATE_GLOBAL_EMA_DECAY,
        reference_samples=config.GATE_REFERENCE_SAMPLES,
        alpha_local_weight=config.GATE_ALPHA_LOCAL_WEIGHT,
        threshold_bias=config.GATE_THRESHOLD_BIAS,
        threshold_quantile=config.THRESHOLD_QUANTILE,
        threshold_mad_scale=config.THRESHOLD_MAD_SCALE,
        threshold_local_update_rate=config.THRESHOLD_LOCAL_UPDATE_RATE,
        threshold_quantile_weight=config.THRESHOLD_QUANTILE_WEIGHT,
        threshold_min_samples_for_optimization=config.THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION,
        score_short_weight=config.GATE_SCORE_SHORT_WEIGHT,
        score_medium_weight=config.GATE_SCORE_MEDIUM_WEIGHT,
        score_long_weight=score_long_weight,
        trigger_hysteresis_margin=config.GATE_TRIGGER_HYSTERESIS_MARGIN,
        min_trigger_interval=getattr(config, "GATE_MIN_TRIGGER_INTERVAL", 1),
        stats_ema_decay=config.STATS_EMA_DECAY,
        stats_window_size=config.STATS_WINDOW_SIZE,
        isolation_update_freq=config.ISOLATION_UPDATE_FREQ,
    )

    if gate_state_path is not None:
        resolved_gate_state_path = Path(gate_state_path)
        if resolved_gate_state_path.exists():
            try:
                gate.load_state(resolved_gate_state_path)
                logger.info(f"已加载预热门控状态: {resolved_gate_state_path}")
            except (ModuleNotFoundError, ValueError, pickle.UnpicklingError) as exc:
                logger.warning(
                    f"门控状态文件不兼容，已回退到冷启动在线门控: {resolved_gate_state_path}; "
                    f"原因: {exc}"
                )
        else:
            logger.warning(f"门控状态文件不存在，使用冷启动在线门控: {resolved_gate_state_path}")

    return gate


def find_nearest_action_index(action_space: np.ndarray, target_value: float) -> int:
    distances = np.abs(action_space - float(target_value))
    return int(np.argmin(distances))


def compute_extended_test_metrics(
    power_values: list[float],
    action_values: list[float],
    action_count: int,
    sample_interval_minutes: float = 5.0,
) -> dict[str, float | int]:
    steps = len(power_values)
    if steps == 0:
        return {
            "duration_days": 0.0,
            "E_total_kwh": 0.0,
            "E_daily_kwh_per_day": 0.0,
            "N_daily_count_per_day": 0.0,
            "sigma_delta_a": 0.0,
        }

    dt_hours = float(sample_interval_minutes) / 60.0
    steps_per_day = max(1.0, 24.0 / dt_hours)
    duration_days = float(steps / steps_per_day)

    power_array = np.asarray(power_values, dtype=np.float64)
    action_array = np.asarray(action_values, dtype=np.float64)
    e_total_kwh = float(np.sum(power_array) * dt_hours)
    e_daily_kwh_per_day = float(e_total_kwh / duration_days) if duration_days > 0 else 0.0
    n_daily = float(action_count / duration_days) if duration_days > 0 else 0.0

    if action_array.size <= 1:
        sigma_delta_a = 0.0
    else:
        sigma_delta_a = float(np.mean(np.abs(np.diff(action_array))))

    return {
        "duration_days": duration_days,
        "E_total_kwh": e_total_kwh,
        "E_daily_kwh_per_day": e_daily_kwh_per_day,
        "N_daily_count_per_day": n_daily,
        "sigma_delta_a": sigma_delta_a,
    }


def get_paper_symbol_field_mapping() -> dict[str, str]:
    return {
        "E_daily": "E_daily_kwh_per_day",
        "eta_saving": "eta_saving_pct",
        "PPR": "PPR_percent",
        "sigma_delta_a": "sigma_delta_a",
        "N_daily": "N_daily_count_per_day",
        "ACR": "ACR",
    }
