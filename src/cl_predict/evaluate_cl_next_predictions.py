"""
比较 cl_next_predictions.csv 中 CL 与 CL_next 的差异，并生成可视化图像。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_prediction_csv(csv_path: Path) -> pd.DataFrame:
    """读取预测结果 CSV 并校验必要列。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"预测结果文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["CL", "CL_next"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少必要列: {missing_cols}，当前列: {df.columns.tolist()}")

    if df.empty:
        raise ValueError(f"CSV 文件为空: {csv_path}")

    return df


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """计算误差指标。"""
    diff = predicted - actual
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs_err = float(np.max(abs_diff))

    non_zero_mask = actual != 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(diff[non_zero_mask] / actual[non_zero_mask])) * 100)
    else:
        mape = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs_err": max_abs_err,
        "mape": mape,
    }


def _plot_comparison(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title_prefix: str = "CL vs CL_next",
):
    """绘制真实值、预测值及误差曲线。"""
    diff = predicted - actual

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(actual, label="CL", linewidth=1.5)
    axes[0].plot(predicted, label="CL_next", linewidth=1.2, alpha=0.9)
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"{title_prefix} - Series")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(diff, label="CL_next - CL", color="tab:red", linewidth=1.2)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Error")
    axes[1].set_title(f"{title_prefix} - Error")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_cl_next_predictions(
    input_csv_path: Path,
    output_img_path: Path,
    start: int | None = None,
    end: int | None = None,
):
    """主流程：读取数据、切片、计算指标、绘图。"""
    df = _load_prediction_csv(input_csv_path)

    if start is None:
        start = 0
    if end is None:
        end = len(df)

    if start < 0 or end < 0 or start >= end or end > len(df):
        raise ValueError(
            f"切片范围非法: start={start}, end={end}, 数据总行数={len(df)}"
        )

    sliced_df = df.iloc[start:end].reset_index(drop=True)
    actual = sliced_df["CL"].to_numpy(dtype=np.float32)
    predicted = sliced_df["CL_next"].to_numpy(dtype=np.float32)

    metrics = _compute_metrics(actual, predicted)
    _plot_comparison(actual, predicted, output_img_path)

    print(f"输入文件: {input_csv_path}")
    print(f"绘图区间: [{start}, {end})，样本数: {len(sliced_df)}")
    print(f"输出图像: {output_img_path}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MaxAbsError: {metrics['max_abs_err']:.6f}")
    if np.isnan(metrics["mape"]):
        print("MAPE: NaN（CL 全为 0，无法计算）")
    else:
        print(f"MAPE: {metrics['mape']:.4f}%")


def main():
    parser = argparse.ArgumentParser(description="比较 CL 与 CL_next 的差异并绘图")
    parser.add_argument(
        "--input_csv_path",
        type=str,
        default=str(Path("data") / "dqn" / "cl_next_predictions.csv"),
        help="输入 CSV 路径，默认 data/dqn/cl_next_predictions.csv",
    )
    parser.add_argument(
        "--output_img_path",
        type=str,
        default=str(Path("data") / "dqn" / "cl_next_predictions_eval.png"),
        help="输出图片路径，默认 data/dqn/cl_next_predictions_eval.png",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="绘图起始下标（含），默认 0",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="绘图结束下标（不含），默认到最后一行",
    )

    args = parser.parse_args()

    evaluate_cl_next_predictions(
        input_csv_path=Path(args.input_csv_path),
        output_img_path=Path(args.output_img_path),
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
