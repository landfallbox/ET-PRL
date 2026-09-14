"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 数据处理工具函数
"""

import pandas as pd


def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    从 DataFrame 中选择指定的列

    参数：
        df: 输入的 DataFrame
        columns: 需要选择的列名列表

    返回：
        包含指定列的新 DataFrame

    异常：
        ValueError: 当指定的列在 DataFrame 中不存在时抛出
    """
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"以下列在 DataFrame 中不存在: {missing_columns}")

    return df[columns].copy()


def split_data(
    df: pd.DataFrame, ratios: list[float], shuffle: bool = True, random_state: int | None = None
) -> list[pd.DataFrame]:
    """
    按指定比例划分 DataFrame 为多个部分

    参数：
        df: 输入的 DataFrame
        ratios: 划分比例列表，每个元素表示对应部分的比例
                比例精度最多支持小数点后两位（如 0.85）
                如果为空列表，直接返回包含原 DataFrame 的列表
                比例总和可小于等于 1.0，如果小于 1.0，剩余数据作为最后一块返回
        shuffle: 是否在划分前打乱数据顺序，默认为 True
        random_state: 随机种子，用于控制打乱的可重复性，默认为 None

    返回：
        划分后的 DataFrame 列表，每个 DataFrame 的索引已重置

    异常：
        ValueError: 当比例包含非正数时抛出
        ValueError: 当比例精度超过小数点后两位时抛出
        ValueError: 当比例总和大于 1.0 时抛出
    """
    if not ratios:
        return [df.reset_index(drop=True)]

    if any(ratio <= 0 or ratio > 1.0 for ratio in ratios):
        raise ValueError("所有比例必须在 (0, 1.0] 范围内")

    ratio_sum = sum(ratios)
    if ratio_sum > 1.0 + 1e-9:
        raise ValueError(f"比例总和不能大于 1.0，当前总和为 {ratio_sum:.10f}")

    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    total_rows = len(df)
    split_indices = []
    current_index = 0

    for ratio in ratios:
        next_index = current_index + int(total_rows * ratio)
        split_indices.append((current_index, next_index))
        current_index = next_index

    if current_index < total_rows:
        split_indices.append((current_index, total_rows))

    result = []
    for start, end in split_indices:
        subset = df.iloc[start:end].reset_index(drop=True)
        result.append(subset)

    return result


def build_temporal_features(
    df: pd.DataFrame, feature_columns: list[str], window_length: int
) -> pd.DataFrame:
    """
    构建时序特征，将之前 n 个时刻的特征拼接到当前行

    参数：
        df: 输入的 DataFrame
        feature_columns: 原始特征列名列表
        window_length: 时序窗口长度（n），表示回溯的时刻数

    返回：
        包含时间维度优先排列的特征列的新 DataFrame，丢弃前 window_length-1 行，索引已重置

    说明：
        列顺序按时间维度优先排列，格式为：
        [t=0时刻的所有特征, t=-1时刻的所有特征, ..., t=-(window_length-1)时刻的所有特征]
    """
    if window_length < 1:
        raise ValueError(f"窗口长度必须大于等于 1，当前值为 {window_length}")

    if not feature_columns:
        raise ValueError("特征列列表不能为空")

    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"以下列在 DataFrame 中不存在: {missing_columns}")

    temp_df = df[feature_columns].copy()

    for col in feature_columns:
        for lag in range(1, window_length):
            lag_col_name = f"{col}_lag{lag}"
            temp_df[lag_col_name] = df[col].shift(lag)

    ordered_columns = []
    for lag in range(window_length):
        for col in feature_columns:
            if lag == 0:
                ordered_columns.append(col)
            else:
                ordered_columns.append(f"{col}_lag{lag}")

    result_df = temp_df[ordered_columns].copy()
    result_df = result_df.iloc[window_length - 1 :].reset_index(drop=True)

    return result_df
