from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any


def coerce_to_existing_type(existing_value: Any, new_value: Any) -> Any:
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
        raise ValueError(f"类型转换失败: 值 {new_value!r} 无法转换为 {target_type_name}") from exc


def load_overrides(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"配置文件必须是 JSON 对象: {config_path}")

    if "best_config_overrides" in payload:
        overrides = payload["best_config_overrides"]
        if not isinstance(overrides, dict):
            raise ValueError(f"best_config_overrides 必须是对象: {config_path}")
        return overrides

    return payload


def apply_overrides(config: Any, overrides: dict[str, Any], source_name: str) -> Any:
    """将覆盖项应用到 frozen dataclass 配置实例，返回新实例。

    旧实现是对配置类 setattr（全局改类属性）；frozen dataclass 不可变，
    故改为基于 dataclasses.replace 生成新实例并显式返回，由调用方传递。
    """
    if not hasattr(config, "__dataclass_fields__"):
        raise TypeError("apply_overrides 需要 dataclass 配置实例")

    valid_keys = {f.name for f in fields(config)}
    updates: dict[str, Any] = {}
    unknown_keys: list[str] = []
    applied_keys: list[str] = []

    for key, value in overrides.items():
        if key not in valid_keys:
            unknown_keys.append(str(key))
            continue

        existing_value = getattr(config, key)
        try:
            coerced_value = coerce_to_existing_type(existing_value, value)
        except ValueError as exc:
            raise ValueError(
                f"[{source_name}] 配置键 {key!r} 类型转换失败，"
                f"当前类型={type(existing_value).__name__}, 输入值={value!r}。{exc}"
            ) from exc

        updates[key] = coerced_value
        applied_keys.append(str(key))

    if unknown_keys:
        print(f"[{source_name}] 忽略未知配置键: {unknown_keys}")
    print(f"[{source_name}] 已应用配置键数量: {len(applied_keys)}")

    return replace(config, **updates) if updates else config
