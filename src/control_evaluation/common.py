from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ml_toolkit.utils import CheckpointManager, Logger, copy_config_snapshot, resolve_experiment_dir

from config.dqn_config import DQNConfig
from src.dqn.agent import DQNAgent
from src.dqn.rewards import RewardCalculator
from src.dqn.trainer import DQNTrainer
from src.online_anomaly_detection.streaming_anomaly_gate import StreamingAnomalyGate


def resolve_train_experiment_dir(train_experiment_dir: Path | None) -> Path:
    return resolve_experiment_dir(
        explicit_dir=train_experiment_dir,
        experiment_name=DQNConfig.EXPERIMENT_NAME,
        mode="train",
        log_root_dir=DQNConfig.LOG_ROOT_DIR,
    )


def copy_train_config(
    resolved_train_dir: Path,
    eval_experiment_dir: Path,
    config: type[DQNConfig],
    logger: Logger,
) -> None:
    copy_config_snapshot(
        source_experiment_dir=resolved_train_dir,
        target_experiment_dir=eval_experiment_dir,
        config_filename=config.CONFIG_FILENAME,
        logger=logger,
    )


def _create_agent(config: type[DQNConfig], action_space: np.ndarray, device: torch.device) -> DQNAgent:
    return DQNAgent(
        state_size=config.STATE_SIZE,
        action_space=action_space,
        hidden_sizes=config.HIDDEN_SIZES,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        epsilon_start=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay=config.EPSILON_DECAY,
        memory_capacity=config.MEMORY_CAPACITY,
        batch_size=config.BATCH_SIZE,
        target_update_freq=config.TARGET_UPDATE_FREQ,
        device=device,
    )


def _create_reward_calculator(
    config: type[DQNConfig],
    test_data: pd.DataFrame,
    action_space: np.ndarray,
) -> RewardCalculator:
    return RewardCalculator(
        data=test_data,
        action_space=action_space,
        weight_efficiency=config.REWARD_WEIGHT_EFFICIENCY,
        weight_comfort=config.REWARD_WEIGHT_COMFORT,
        target_supply_temp=config.TARGET_SUPPLY_TEMP,
        coeff_path=config.COEFF_DATE_PATH,
        chiller_capacity=config.CHILLER_CAPACITY,
        chiller_ref_power=config.CHILLER_REF_POWER,
        supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
        comfort_sigma=config.COMFORT_SIGMA,
        chiller_high_threshold=config.CHILLER_HIGH_THRESHOLD,
        chiller_medium_threshold=config.CHILLER_MEDIUM_THRESHOLD,
        chiller_low_threshold=config.CHILLER_LOW_THRESHOLD,
        f_nominal=config.CHILLER_F_NOMINAL,
        f_cw=config.CHILLER_F_CW,
        f_tower=config.CHILLER_F_TOWER,
        f_chw=config.CHILLER_F_CHW,
        c_p=config.CHILLER_CP,
        density_water=config.CHILLER_WATER_DENSITY,
    )


def build_eval_components(
    config: type[DQNConfig],
    resolved_train_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, DQNAgent, dict, RewardCalculator]:
    test_data = DQNTrainer._load_data(config.get_test_data_path(), config.STATE_COLUMNS)
    if test_data.empty:
        raise ValueError(f"测试数据为空: {config.get_test_data_path()}")

    action_space = DQNTrainer._load_action_space(config.ACTION_SPACE_PATH)
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

    reward_calc = _create_reward_calculator(config=config, test_data=test_data, action_space=action_space)

    return test_data, action_space, agent, checkpoint, reward_calc


def create_streaming_gate(
    config: type[Any],
    test_data: pd.DataFrame,
    logger: Logger,
    gate_state_path: Path | None,
) -> StreamingAnomalyGate:
    feature_columns = config.FEATURE_COLUMNS
    missing_columns = [column for column in feature_columns if column not in test_data.columns]
    if missing_columns:
        raise ValueError(f"测试数据缺少异常门控特征列: {missing_columns}")

    gate = StreamingAnomalyGate(
        feature_dim=len(feature_columns),
        local_window_size=config.GATE_LOCAL_WINDOW_SIZE,
        global_ema_decay=config.GATE_GLOBAL_EMA_DECAY,
        reference_samples=config.GATE_REFERENCE_SAMPLES,
        contamination=config.GATE_CONTAMINATION,
        alpha_local_weight=config.GATE_ALPHA_LOCAL_WEIGHT,
        threshold_bias=config.GATE_THRESHOLD_BIAS,
        threshold_quantile=config.THRESHOLD_QUANTILE,
        threshold_mad_scale=config.THRESHOLD_MAD_SCALE,
        threshold_local_update_rate=config.THRESHOLD_LOCAL_UPDATE_RATE,
        threshold_quantile_weight=config.THRESHOLD_QUANTILE_WEIGHT,
        score_short_weight=config.GATE_SCORE_SHORT_WEIGHT,
        score_medium_weight=config.GATE_SCORE_MEDIUM_WEIGHT,
        score_long_weight=config.GATE_SCORE_LONG_WEIGHT,
    )

    if gate_state_path is not None:
        resolved_gate_state_path = Path(gate_state_path)
        if resolved_gate_state_path.exists():
            gate.load_state(resolved_gate_state_path)
            logger.info(f"已加载预热门控状态: {resolved_gate_state_path}")
        else:
            logger.warning(f"门控状态文件不存在，使用冷启动在线门控: {resolved_gate_state_path}")

    return gate


def find_nearest_action_index(action_space: np.ndarray, target_value: float) -> int:
    distances = np.abs(action_space - float(target_value))
    return int(np.argmin(distances))


def compute_violation_metrics(
    chiller_supply_temperature: list[float],
    comfort_lower_bound: float = 15.0,
    comfort_upper_bound: float = 19.0,
) -> tuple[int, float]:
    if not chiller_supply_temperature:
        return 0, 0.0

    violation_count = sum(
        1
        for temperature in chiller_supply_temperature
        if temperature < comfort_lower_bound or temperature > comfort_upper_bound
    )
    violation_rate = violation_count / len(chiller_supply_temperature)
    return int(violation_count), float(violation_rate * 100.0)
