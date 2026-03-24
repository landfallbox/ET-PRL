import argparse
import json
from pathlib import Path
from typing import Any

from config.dqn_config import DQNConfig
from config.online_anomaly_detection_config import OnlineAnomalyDetectionConfig
from src.online_anomaly_detection.prewarm_streaming_gate import prewarm_gate


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


def _apply_overrides_to_gate_config(
    gate_config: type[OnlineAnomalyDetectionConfig],
    overrides: dict[str, Any],
    source_name: str,
) -> None:
    unknown_keys: list[str] = []
    applied_keys: list[str] = []

    for key, value in overrides.items():
        if not hasattr(gate_config, key):
            unknown_keys.append(str(key))
            continue

        existing_value = getattr(gate_config, key)
        try:
            coerced_value = _coerce_with_existing_type(existing_value, value)
        except ValueError as exc:
            raise ValueError(
                f"[{source_name}] 配置键 {key!r} 类型转换失败，"
                f"当前类型={type(existing_value).__name__}, 输入值={value!r}。{exc}"
            ) from exc

        setattr(gate_config, key, coerced_value)
        applied_keys.append(str(key))

    if unknown_keys:
        print(f"[{source_name}] 忽略未知配置键: {unknown_keys}")
    print(f"[{source_name}] 已应用配置键数量: {len(applied_keys)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="预热 StreamingAnomalyGate 并保存状态")
    parser.add_argument(
        "--gate_config_path",
        type=str,
        default=None,
        help="门控超参配置 JSON 路径（支持包含 best_config_overrides 的优化输出）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DQNConfig.get_data_dir() / "streaming_anomaly_gate_state.pkl"),
    )
    args = parser.parse_args()

    if args.gate_config_path:
        gate_config_path = Path(args.gate_config_path)
        gate_overrides = _load_overrides(gate_config_path)
        _apply_overrides_to_gate_config(
            OnlineAnomalyDetectionConfig,
            gate_overrides,
            source_name="GATE",
        )

    prewarm_gate(Path(args.output))
