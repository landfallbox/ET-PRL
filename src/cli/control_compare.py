import argparse
from pathlib import Path

from config.control_compare_config import ControlCompareConfig
from config.dqn_config import DQNConfig
from ml_toolkit.utils import CheckpointManager
from src.cli.overrides import apply_overrides, load_overrides
from src.comparison.compare_control_strategies import compare_control_strategies


def _resolve_dqn_model_path(dqn_model_arg: str | None) -> Path:
    if dqn_model_arg:
        explicit_model_path = Path(dqn_model_arg)
        if not explicit_model_path.exists():
            raise FileNotFoundError(f"指定的 DQN 模型文件不存在: {explicit_model_path}")
        return explicit_model_path

    latest_train_experiment = CheckpointManager.find_latest_experiment(
        experiment_name=DQNConfig.EXPERIMENT_NAME,
        mode="train",
        log_root_dir=ControlCompareConfig.LOG_ROOT_DIR,
    )
    if latest_train_experiment is None:
        raise FileNotFoundError(
            "未提供 --dqn_model，且未找到任何 DQN 训练实验目录。"
            f"请检查目录: {ControlCompareConfig.LOG_ROOT_DIR / DQNConfig.EXPERIMENT_NAME / ControlCompareConfig.TRAIN_SUBDIR}"
        )

    best_model_path = (
        latest_train_experiment
        / ControlCompareConfig.CHECKPOINT_DIR_NAME
        / ControlCompareConfig.BEST_MODEL_FILENAME
    )
    if not best_model_path.exists():
        raise FileNotFoundError(
            "未提供 --dqn_model，自动选择最新实验失败：best model 不存在。"
            f"期望路径: {best_model_path}"
        )

    print(f"使用最新实验 best model: {best_model_path}")
    return best_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="控制策略性能对比")
    parser.add_argument("--dqn_model", type=str, default=None, help="DQN 训练模型的 .pth 文件路径")
    parser.add_argument(
        "--fixed_interval",
        type=int,
        nargs="+",
        default=[1],
        help="固定间隔基线策略的动作更新间隔，支持多个值进行对比",
    )
    parser.add_argument(
        "--gate_config_path",
        type=str,
        default=None,
        help="门控超参配置 JSON 路径（支持包含 best_config_overrides 的优化输出）",
    )
    parser.add_argument(
        "--gate_state_path",
        type=str,
        default=str(ControlCompareConfig.get_data_dir() / "streaming_anomaly_gate_state.pkl"),
        help="门控状态文件路径",
    )
    parser.add_argument(
        "--no-load-gate-state",
        action="store_true",
        help="跳过加载预热门控状态，从冷启动开始评估",
    )
    args = parser.parse_args()

    if args.gate_config_path:
        gate_config_path = Path(args.gate_config_path)
        gate_overrides = load_overrides(gate_config_path)
        apply_overrides(ControlCompareConfig, gate_overrides, source_name="GATE")

    resolved_dqn_model_path = _resolve_dqn_model_path(args.dqn_model)
    gate_state_path = None if args.no_load_gate_state else (Path(args.gate_state_path) if args.gate_state_path else None)

    compare_control_strategies(
        dqn_model_path=resolved_dqn_model_path,
        fixed_intervals=[int(i) for i in args.fixed_interval],
        gate_state_path=gate_state_path,
    )
