import argparse
from pathlib import Path

from config.lstm_config import LSTMConfig
from src.cl_predict.cl_next_predict import predict_cl_next


def main() -> None:
    parser = argparse.ArgumentParser(description="使用最新 LSTM 模型预测 raw_data.csv 的 CL_next")
    parser.add_argument("--raw_data_path", type=str, default=str(LSTMConfig.RAW_DATA_PATH))
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default=str(Path("data") / "dqn" / "cl_next_predictions.csv"),
    )
    parser.add_argument("--experiment_dir", type=str, default=None)
    args = parser.parse_args()

    predict_cl_next(
        raw_data_path=Path(args.raw_data_path),
        output_csv_path=Path(args.output_csv_path),
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
    )
