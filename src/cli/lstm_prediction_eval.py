import argparse
from pathlib import Path

from src.cl_predict.evaluate_cl_next_predictions import evaluate_cl_next_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 CL 与 CL_next 的差异并绘图")
    parser.add_argument(
        "--input_csv_path",
        type=str,
        default=str(Path("data") / "dqn" / "cl_next_predictions.csv"),
    )
    parser.add_argument(
        "--output_img_path",
        type=str,
        default=str(Path("data") / "dqn" / "cl_next_predictions_eval.png"),
    )
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    evaluate_cl_next_predictions(
        input_csv_path=Path(args.input_csv_path),
        output_img_path=Path(args.output_img_path),
        start=args.start,
        end=args.end,
    )
