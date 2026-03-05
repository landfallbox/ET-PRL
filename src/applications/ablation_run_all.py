import argparse
from pathlib import Path

from config.compare_dqn_config import CompareDQNConfig
from src.ablation.full_et_prl import run_full_et_prl_ablation
from src.ablation.wo_dual_threshold import run_wo_dual_threshold_ablation
from src.ablation.wo_multi_scale import run_wo_multi_scale_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="一键运行全部消融实验")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--fixed_interval", type=int, default=1)
    parser.add_argument(
        "--gate_state_path",
        type=str,
        default=str(CompareDQNConfig.get_data_dir() / "streaming_anomaly_gate_state.pkl"),
    )
    args = parser.parse_args()

    train_experiment_dir = Path(args.experiment_dir) if args.experiment_dir else None
    gate_state_path = Path(args.gate_state_path) if args.gate_state_path else None
    fixed_interval = int(args.fixed_interval)

    run_wo_dual_threshold_ablation(
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
    run_wo_multi_scale_ablation(
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
    run_full_et_prl_ablation(
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )


if __name__ == "__main__":
    main()
