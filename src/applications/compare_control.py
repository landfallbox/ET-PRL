import argparse
import json
from pathlib import Path
from typing import Any

from config.compare_dqn_config import CompareDQNConfig
from src.comparison.compare_control_strategies import compare_control_strategies


def _coerce_with_existing_type(existing_value: Any, new_value: Any) -> Any:
    target_type_name = type(existing_value).__name__

    try:
        if isinstance(existing_value, bool):
            if isinstance(new_value, str):
                normalized = new_value.strip().lower()
                if normalized in {"1", "true", "yes", "y", "on"}:
                    return True
                if normalized in {"0", "false", "no", "n", "off"}:
                    return False
                raise ValueError(f"无法解析布尔值: {new_value!r}")
            return bool(new_value)

        if isinstance(existing_value, int) and not isinstance(existing_value, bool):
            return int(new_value)

        if isinstance(existing_value, float):
            return float(new_value)

        if isinstance(existing_value, list):
            if isinstance(new_value, list):
                return new_value
            raise ValueError(f"期望 list，实际为 {type(new_value).__name__}")

        if isinstance(existing_value, tuple):
            if isinstance(new_value, (list, tuple)):
                return tuple(new_value)
            raise ValueError(f"期望 tuple/list，实际为 {type(new_value).__name__}")

        return new_value
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"类型转换失败: 值 {new_value!r} 无法转换为 {target_type_name}"
        ) from exc


def _load_overrides(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"配置文件必须是 JSON 对象: {config_path}")

    if "best_config_overrides" in payload:
        overrides = payload["best_config_overrides"]
        if not isinstance(overrides, dict):
            raise ValueError(f"best_config_overrides 必须是对象: {config_path}")
        return overrides

    return payload


def _apply_overrides_to_compare_config(
    compare_config: type[CompareDQNConfig],
    overrides: dict[str, Any],
    source_name: str,
) -> None:
    unknown_keys: list[str] = []
    applied_keys: list[str] = []

    for key, value in overrides.items():
        if not hasattr(compare_config, key):
            unknown_keys.append(str(key))
            continue

        existing_value = getattr(compare_config, key)
        try:
            coerced_value = _coerce_with_existing_type(existing_value, value)
        except ValueError as exc:
            raise ValueError(
                f"[{source_name}] 配置键 {key!r} 类型转换失败，"
                f"当前类型={type(existing_value).__name__}, 输入值={value!r}。{exc}"
            ) from exc

        setattr(compare_config, key, coerced_value)
        applied_keys.append(str(key))

    if unknown_keys:
        print(f"[{source_name}] 忽略未知配置键: {unknown_keys}")
    print(f"[{source_name}] 已应用配置键数量: {len(applied_keys)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="控制策略性能对比")
    parser.add_argument("--experiment_dir", type=str, default=None, help="DQN 训练实验目录（如 logs/dqn/train/<timestamp>）")
    parser.add_argument("--fixed_interval", type=int, default=1, help="固定间隔基线策略的动作更新间隔")
    parser.add_argument(
        "--gate_config_path",
        type=str,
        default=None,
        help="门控超参配置 JSON 路径（支持包含 best_config_overrides 的优化输出）",
    )
    parser.add_argument(
        "--gate_state_path",
        type=str,
        default=str(CompareDQNConfig.get_data_dir() / "streaming_anomaly_gate_state.pkl"),
        help="门控状态文件路径",
    )
    parser.add_argument(
        "--no-load-gate-state",
        action="store_true",
        help="跳过加载预热门控状态，从冷启动开始评估（用于新超参的精确性能评估）",
    )
    args = parser.parse_args()

    if args.gate_config_path:
        gate_config_path = Path(args.gate_config_path)
        gate_overrides = _load_overrides(gate_config_path)
        _apply_overrides_to_compare_config(CompareDQNConfig, gate_overrides, source_name="GATE")

    # 如果指定了 --no-load-gate-state，则传 None 给 gate_state_path
    gate_state_path = None if args.no_load_gate_state else (Path(args.gate_state_path) if args.gate_state_path else None)

    compare_control_strategies(
        train_experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        fixed_interval=int(args.fixed_interval),
        gate_state_path=gate_state_path,
    )


if __name__ == "__main__":
    main()
