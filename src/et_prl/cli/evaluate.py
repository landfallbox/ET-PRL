import argparse
from pathlib import Path

from et_prl.config.loader import get_default
from et_prl.evaluation.lstm_eval import run_lstm
from et_prl.evaluation.dqn_evaluator import run_dqn
from et_prl.evaluation.event_driven_eval import run_event_driven


def main_dqn() -> None:
    parser = argparse.ArgumentParser(description="固定间隔控制策略测试")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--fixed_interval", type=int, default=4)
    args = parser.parse_args()

    run_dqn(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
    )


def main_lstm() -> None:
    parser = argparse.ArgumentParser(description="LSTM 模型测试")
    parser.add_argument("--experiment_dir", type=str, default=None)
    args = parser.parse_args()
    run_lstm(train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None)


def main_event_driven() -> None:
    parser = argparse.ArgumentParser(description="事件驱动控制策略测试")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument(
        "--gate_state_path",
        type=str,
        default=str(get_default("event_driven").get_data_dir() / "streaming_anomaly_gate_state.pkl"),
    )
    args = parser.parse_args()

    run_event_driven(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )
