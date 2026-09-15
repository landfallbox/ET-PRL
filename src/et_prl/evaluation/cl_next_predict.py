"""
使用最新 LSTM 训练实验对 raw_data.csv 进行 CL_next 预测
"""
import json
from pathlib import Path

import pandas as pd
import torch

from et_prl.config.loader import load_config
from et_prl.data import tensor_loader
from et_prl.models import LSTM
from et_prl.utils import CheckpointManager, ConfigManager


BUILD_SLIDING_WINDOW_SEQUENCES = getattr(tensor_loader, "build_sliding_window_sequences")


def _zscore_transform(df: pd.DataFrame, mean_dict: dict, std_dict: dict, columns: list[str]) -> pd.DataFrame:
    """对指定列执行 z-score 标准化。"""
    result_df = df.copy()
    for col in columns:
        if col not in result_df.columns:
            raise ValueError(f"输入数据缺少特征列: {col}")
        col_mean = float(mean_dict[col])
        col_std = float(std_dict[col])
        if col_std == 0:
            raise ValueError(f"归一化参数中列 {col} 的标准差为 0，无法标准化")
        result_df[col] = (result_df[col].astype(float) - col_mean) / col_std
    return result_df


def _build_sequences(feature_array, window_length: int):
    """构建 LSTM 输入序列，返回形状 (N-window, window, feature_dim)。"""
    feature_tensor = torch.as_tensor(feature_array, dtype=torch.float32)
    num_rows = feature_tensor.shape[0]
    if num_rows <= window_length:
        raise ValueError(
            f"样本数量不足，当前行数={num_rows}，窗口长度={window_length}，至少需要 {window_length + 1} 行"
        )

    sequence_features = BUILD_SLIDING_WINDOW_SEQUENCES(feature_tensor, window_length)
    return sequence_features[:-1]


def _load_normalizer(normalizer_path: Path) -> dict:
    """读取 normalizer.json。"""
    if not normalizer_path.exists():
        raise FileNotFoundError(f"未找到归一化参数文件: {normalizer_path}")

    with normalizer_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def predict_cl_next(
    raw_data_path: Path,
    output_csv_path: Path,
    train_experiment_dir: Path | None = None,
):
    """
    使用最新一次 LSTM 训练实验模型预测下一时刻 CL。

    输出 CSV 列: CL, Twb, CL_next
    """
    base_config = load_config("lstm")

    if train_experiment_dir is None:
        train_experiment_dir = CheckpointManager.find_latest_experiment("lstm", mode="train")
        if train_experiment_dir is None:
            raise FileNotFoundError("未找到任何 LSTM 训练实验目录")
    else:
        train_experiment_dir = Path(train_experiment_dir)
        if not train_experiment_dir.exists():
            raise FileNotFoundError(f"指定训练实验目录不存在: {train_experiment_dir}")

    train_config = ConfigManager(train_experiment_dir).load_config()

    feature_columns = train_config.get("FEATURE_COLUMNS", base_config.FEATURE_COLUMNS)
    window_length = int(train_config.get("WINDOW_LENGTH", base_config.WINDOW_LENGTH))
    input_size = int(train_config.get("INPUT_SIZE", base_config.INPUT_SIZE))
    hidden_sizes = train_config.get("HIDDEN_SIZES", base_config.HIDDEN_SIZES)
    output_size = int(train_config.get("OUTPUT_SIZE", base_config.OUTPUT_SIZE))
    batch_first = bool(train_config.get("BATCH_FIRST", base_config.BATCH_FIRST))
    dropout = float(train_config.get("DROPOUT", base_config.DROPOUT))

    if len(feature_columns) != input_size:
        raise ValueError(
            f"配置不一致：FEATURE_COLUMNS 长度={len(feature_columns)}，INPUT_SIZE={input_size}"
        )

    raw_df = pd.read_csv(raw_data_path)
    for col in feature_columns:
        if col not in raw_df.columns:
            raise ValueError(f"raw_data.csv 缺少必要列: {col}")

    normalizer = _load_normalizer(base_config.get_normalizer_path())
    feature_mean = normalizer["feature"]["mean"]
    feature_std = normalizer["feature"]["std"]

    target_key = f"{base_config.TARGET_COLUMN}_target"
    target_mean = float(normalizer["target"]["mean"][target_key])
    target_std = float(normalizer["target"]["std"][target_key])
    cl_mean = float(feature_mean[base_config.TARGET_COLUMN])
    cl_std = float(feature_std[base_config.TARGET_COLUMN])

    scaled_df = _zscore_transform(raw_df, feature_mean, feature_std, feature_columns)
    features_scaled = scaled_df[feature_columns].to_numpy(dtype="float32")
    sequences = _build_sequences(features_scaled, window_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        batch_first=batch_first,
        dropout=dropout,
    ).to(device)

    CheckpointManager.load_best_model(train_experiment_dir, model, map_location=device)
    model.eval()

    with torch.no_grad():
        y_pred = model(sequences.to(device)).squeeze(-1).detach().cpu().numpy()

    y_pred_target = y_pred * target_std + target_mean
    y_pred_cl = y_pred_target * cl_std + cl_mean

    aligned_df = raw_df.iloc[window_length - 1:-1].copy()
    result_df = pd.DataFrame({
        "CL": aligned_df["CL"].to_numpy(),
        "Twb": aligned_df["Twb"].to_numpy(),
        "CL_next": y_pred_cl,
    })

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)

    print(f"使用训练实验目录: {train_experiment_dir}")
    print(f"输入数据: {raw_data_path}")
    print(f"输出文件: {output_csv_path}")
    print(f"输出行数: {len(result_df)}")
