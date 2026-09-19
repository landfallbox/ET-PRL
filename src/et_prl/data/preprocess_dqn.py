"""DQN 数据划分脚本。"""

from pathlib import Path

import pandas as pd

from et_prl.config.loader import get_default, load_config
from et_prl.data import split_data


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r <= 0 or r >= 1 for r in ratios):
        raise ValueError("训练/验证/测试比例必须在 (0, 1) 区间")

    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"训练/验证/测试比例之和必须为 1.0，当前为 {total:.10f}")


def _prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_input_columns = ["CL", "Twb"]
    missing = [col for col in required_input_columns if col not in df.columns]
    if missing:
        raise ValueError(f"输入数据缺少必要列: {missing}")

    if "CL_predict" not in df.columns:
        if "CL_next" in df.columns:
            df = df.rename(columns={"CL_next": "CL_predict"})
        else:
            raise ValueError("输入数据缺少预测列，需要 CL_predict 或 CL_next")

    output_columns = get_default("dqn").STATE_COLUMNS
    missing_output_cols = [col for col in output_columns if col not in df.columns]
    if missing_output_cols:
        raise ValueError(f"处理后数据仍缺少状态列: {missing_output_cols}")

    return df[output_columns].copy()


def split_dqn_dataset() -> None:
    config = load_config("dqn")
    input_path = config.get_data_dir() / "cl_next_predictions.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    _validate_split_ratios(config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO)

    print(f"读取输入数据: {input_path}")
    raw_data = pd.read_csv(input_path)
    print(f"输入数据形状: {raw_data.shape}")

    prepared_data = _prepare_columns(raw_data)
    print(f"用于 DQN 的状态列: {prepared_data.columns.tolist()}")

    split_result = split_data(
        prepared_data,
        ratios=[config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO],
        shuffle=config.SHUFFLE_DATA,
        random_state=config.RANDOM_STATE,
    )

    if len(split_result) == 3:
        train_data, val_data, test_data = split_result
    elif len(split_result) == 4:
        train_data, val_data, test_data, remain_data = split_result
        test_data = pd.concat([test_data, remain_data], ignore_index=True)
    else:
        raise ValueError(f"划分结果数量异常，期望 3 或 4 段，实际为 {len(split_result)}")

    output_dir: Path = config.get_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = config.get_train_data_path()
    val_path = config.get_val_data_path()
    test_path = config.get_test_data_path()

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"训练集形状: {train_data.shape} -> {train_path}")
    print(f"验证集形状: {val_data.shape} -> {val_path}")
    print(f"测试集形状: {test_data.shape} -> {test_path}")
    print("DQN 数据划分完成")


def main() -> None:
    """CLI 入口函数"""
    split_dqn_dataset()
