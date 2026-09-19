"""lstm 数据预处理。"""

import pandas as pd

from et_prl.config.loader import load_config
from et_prl.data import Normalizer, build_temporal_features, select_columns, split_data


def add_target_column(df: pd.DataFrame, target_column: str, target_shift: int = -1) -> pd.DataFrame:
    """
    为数据添加目标列，并删除无法获取目标值的行

    参数：
        df: 输入的 DataFrame，应为已构建时序特征的数据
        target_column: 目标变量的原始列名
        target_shift: 目标值相对于当前行的偏移量，默认为 -1 表示下一时刻

    返回：
        包含目标列的新 DataFrame，已删除无法获取目标值的行，索引已重置

    说明：
        target_shift = -1: 目标值为下一时刻（shift(-1)）
        target_shift = 0: 目标值为当前时刻
        target_shift = 1: 目标值为上一时刻（shift(1)）
    """
    if target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 在 DataFrame 中不存在")

    result_df = df.copy()

    # 添加目标列
    target_col_name = f"{target_column}_target"
    result_df[target_col_name] = df[target_column].shift(target_shift)

    # 删除包含 NaN 目标值的行
    result_df = result_df.dropna(subset=[target_col_name]).reset_index(drop=True)

    return result_df


def preprocess_data():
    """
    LSTM 数据预处理主函数

    流程：
    1. 从配置读取原始数据路径和特征列
    2. 加载原始数据
    3. 提取指定的特征列
    4. 按比例划分数据集（训练、验证、测试）
    5. 归一化（基于训练集计算参数，应用到所有数据集）
    6. 构建时序特征（滑动窗口特征工程）
    7. 添加目标列（下一时刻的目标值）
    8. 保存划分后的数据到指定目录
    """
    # 获取配置
    config = load_config("lstm")

    # 1. 加载原始数据
    print(f"加载原始数据：{config.RAW_DATA_PATH}")
    raw_data = pd.read_csv(
        config.RAW_DATA_PATH,
    )
    print(f"原始数据形状：{raw_data.shape}")
    print(f"数据列名：{raw_data.columns.tolist()}")

    # 2. 提取特征列
    print(f"提取特征列：{config.FEATURE_COLUMNS}")
    feature_data = select_columns(raw_data, config.FEATURE_COLUMNS)
    print(f"特征数据形状：{feature_data.shape}")

    # 3. 划分数据集
    print("按比例划分数据集")
    print(f"训练集比例：{config.TRAIN_RATIO}")
    print(f"验证集比例：{config.VAL_RATIO}")
    print(f"测试集比例：{config.TEST_RATIO}")

    split_ratios = [config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO]
    data_split = split_data(
        feature_data,
        ratios=split_ratios,
        shuffle=config.SHUFFLE_DATA,
        random_state=config.RANDOM_STATE,
    )

    train_data, val_data, test_data = data_split[0], data_split[1], data_split[2]
    print(f"训练集形状：{train_data.shape}")
    print(f"验证集形状：{val_data.shape}")
    print(f"测试集形状：{test_data.shape}")

    # 4. 创建数据保存目录
    data_dir = config.get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存数据到目录：{data_dir}")

    # 5. 归一化处理（基于训练集计算参数，应用到所有数据集）
    print("执行数据归一化...")
    feature_normalizer = Normalizer()
    feature_normalizer.fit(train_data, config.FEATURE_COLUMNS)

    # 对所有数据集应用特征归一化
    train_data = feature_normalizer.transform(train_data, config.FEATURE_COLUMNS)
    val_data = feature_normalizer.transform(val_data, config.FEATURE_COLUMNS)
    test_data = feature_normalizer.transform(test_data, config.FEATURE_COLUMNS)

    print("特征数据归一化完成")

    # 6. 构建时序特征
    print(f"构建时序特征，窗口长度：{config.WINDOW_LENGTH}")
    train_data = build_temporal_features(train_data, config.FEATURE_COLUMNS, config.WINDOW_LENGTH)
    val_data = build_temporal_features(val_data, config.FEATURE_COLUMNS, config.WINDOW_LENGTH)
    test_data = build_temporal_features(test_data, config.FEATURE_COLUMNS, config.WINDOW_LENGTH)
    print("时序特征构建完成")
    print(f"训练集形状（构建特征后）：{train_data.shape}")
    print(f"验证集形状（构建特征后）：{val_data.shape}")
    print(f"测试集形状（构建特征后）：{test_data.shape}")

    # 7. 添加目标列
    print(f"添加目标列：{config.TARGET_COLUMN}")
    train_data = add_target_column(train_data, config.TARGET_COLUMN)
    val_data = add_target_column(val_data, config.TARGET_COLUMN)
    test_data = add_target_column(test_data, config.TARGET_COLUMN)
    print("目标列添加完成")
    print(f"训练集形状（添加目标列后）：{train_data.shape}")
    print(f"验证集形状（添加目标列后）：{val_data.shape}")
    print(f"测试集形状（添加目标列后）：{test_data.shape}")

    # 8. 归一化目标列（基于训练集计算参数）
    print("执行目标列归一化...")
    target_col = f"{config.TARGET_COLUMN}_target"
    target_normalizer = Normalizer()
    target_normalizer.fit(train_data, [target_col])

    # 对所有数据集应用目标归一化
    train_data = target_normalizer.transform(train_data, [target_col])
    val_data = target_normalizer.transform(val_data, [target_col])
    test_data = target_normalizer.transform(test_data, [target_col])

    print("目标列归一化完成")

    # 9. 保存归一化参数到单个文件
    normalizer_path = config.get_normalizer_path()
    Normalizer.save_normalizers(
        {"feature": feature_normalizer, "target": target_normalizer}, normalizer_path
    )
    print(f"归一化参数已保存：{normalizer_path}")

    # 10. 保存划分后的数据
    train_path = config.get_train_data_path()
    val_path = config.get_val_data_path()
    test_path = config.get_test_data_path()

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"训练集已保存：{train_path}")
    print(f"验证集已保存：{val_path}")
    print(f"测试集已保存：{test_path}")
    print("数据预处理完成！")


def main() -> None:
    """CLI 入口函数"""
    preprocess_data()
