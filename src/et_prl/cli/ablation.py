import argparse
from pathlib import Path

from et_prl.experiments.ablation.full_et_prl import run_full_et_prl_ablation
from et_prl.experiments.ablation.wo_dual_threshold import run_wo_dual_threshold_ablation
from et_prl.experiments.ablation.wo_local_threshold import run_wo_local_threshold_ablation
from et_prl.experiments.ablation.wo_multi_scale import (
    run_single_scale_long_ablation,
    run_single_scale_medium_ablation,
    run_single_scale_short_ablation,
    run_wo_multi_scale_ablation,
)


def _parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--fixed_interval", type=int, default=1)
    parser.add_argument(
        "--gate_state_path",
        type=str,
        default=None,
        help="门控状态文件路径；默认不加载（冷启动）",
    )
    return parser.parse_args()


def main_all() -> None:
    args = _parse_common_args("一键运行全部消融实验")
    train_experiment_dir = Path(args.experiment_dir) if args.experiment_dir else None
    gate_state_path = Path(args.gate_state_path) if args.gate_state_path else None
    fixed_interval = int(args.fixed_interval)

    run_wo_dual_threshold_ablation(
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
    run_wo_local_threshold_ablation(
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
    run_single_scale_short_ablation(
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
    run_single_scale_medium_ablation(
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
    run_single_scale_long_ablation(
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )
    run_full_et_prl_ablation(
        train_experiment_dir=train_experiment_dir,
        fixed_interval=fixed_interval,
        gate_state_path=gate_state_path,
    )


def main_full_et_prl() -> None:
    args = _parse_common_args("消融实验：Full ET-PRL")
    run_full_et_prl_ablation(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )


def main_wo_dual_threshold() -> None:
    args = _parse_common_args("消融实验：Without Global Threshold (Local-Only)")
    run_wo_dual_threshold_ablation(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )


def main_wo_local_threshold() -> None:
    args = _parse_common_args("消融实验：Without Local Threshold (Global-Only)")
    run_wo_local_threshold_ablation(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )


def main_wo_multi_scale() -> None:
    args = _parse_common_args("消融实验：w/o Multi-scale")
    run_wo_multi_scale_ablation(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )


def main_single_scale_short() -> None:
    args = _parse_common_args("消融实验：Without Medium/Long Scales (Short-Only)")
    run_single_scale_short_ablation(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )


def main_single_scale_medium() -> None:
    args = _parse_common_args("消融实验：Without Short/Long Scales (Medium-Only)")
    run_single_scale_medium_ablation(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )


def main_single_scale_long() -> None:
    args = _parse_common_args("消融实验：Without Short/Medium Scales (Long-Only)")
    run_single_scale_long_ablation(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
        gate_state_path=Path(args.gate_state_path) if args.gate_state_path else None,
    )
