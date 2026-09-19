"""
配置加载器：YAML 为准 + dataclass schema 严格校验。

用法：
    from et_prl.config.loader import load_config, load_config_path
    cfg = load_config("dqn")            # 加载 configs/dqn.yaml -> DQNConfig
    cfg = load_config_path("my.yaml")   # 加载任意 YAML 路径

规则（见 MIGRATION_PLAN.md 2.4）：
- YAML 是单一事实来源；字段缺失或出现未知键时立即报错（fail-fast）。
- extends 支持单个预设名或预设名列表，按顺序深合并（后者覆盖前者）。
- 路径字段（Path 注解）的相对路径解析为项目根绝对路径。
- TIMESTAMP 由 loader 注入当前时间戳；运行期用 dataclasses.replace 覆盖。
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from et_prl.config.base import project_root
from et_prl.config.control_compare import ControlCompareConfig
from et_prl.config.dqn import DQNConfig
from et_prl.config.event_driven_dqn import EventDrivenDQNConfig
from et_prl.config.gate import GateConfig
from et_prl.config.lstm import LSTMConfig

_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"

# 预设名 -> dataclass schema
PRESET_MAP: dict[str, type] = {
    "lstm": LSTMConfig,
    "dqn": DQNConfig,
    "gate": GateConfig,
    "control_compare": ControlCompareConfig,
    "event_driven": EventDrivenDQNConfig,
}


class ConfigError(ValueError):
    """配置加载/校验错误。"""


def _configs_file(name: str) -> Path:
    return _CONFIGS_DIR / f"{name}.yaml"


def _load_yaml_file(path: Path, _ancestors: set[str]) -> dict[str, Any]:
    """加载并合并单个 YAML。_ancestors 为当前祖先链（用于环检测）。

    菱形继承（同一文件被多个分支 extends）是合法的，故环检测只看当前分支链，
    而非全局已访问集合。
    """
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    key = str(path.resolve())
    if key in _ancestors:
        raise ConfigError(f"检测到 extends 循环引用: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射: {path}")
    extends = data.pop("extends", None)
    merged: dict[str, Any] = {}
    if extends is not None:
        names = [extends] if isinstance(extends, str) else list(extends)
        branch = _ancestors | {key}
        for name in names:
            if not isinstance(name, str):
                raise ConfigError(f"extends 项必须是字符串: {name!r}")
            merged.update(_load_yaml_file(_configs_file(name), branch))
    merged.update(data)
    return merged


def _resolve_path(value: Any, root: Path) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else (root / p).resolve()


def _convert_value(field, value: Any, root: Path) -> Any:
    # dataclass 文件使用 `from __future__ import annotations`，field.type 为字符串
    hint = field.type if isinstance(field.type, str) else getattr(field.type, "__name__", "")
    if hint == "Path":
        return _resolve_path(value, root)
    if hint == "tuple":
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"字段 {field.name} 应为列表/元组: {value!r}")
        return tuple(value)
    return value


def _validate_and_build(
    schema: type, merged: dict[str, Any], root: Path, timestamp: str | None = None
):
    if not is_dataclass(schema):
        raise ConfigError(f"{schema} 不是 dataclass，无法作为配置 schema")
    schema_fields = {f.name: f for f in fields(schema)}

    # TIMESTAMP 由 loader 注入（YAML 可显式提供以复现实验），须在缺失检查前
    if "TIMESTAMP" in schema_fields and "TIMESTAMP" not in merged:
        merged["TIMESTAMP"] = (
            timestamp if timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    missing = [n for n in schema_fields if n not in merged]
    if missing:
        raise ConfigError(f"配置缺失字段（schema={schema.__name__}）: {', '.join(sorted(missing))}")
    unknown = [k for k in merged if k not in schema_fields]
    if unknown:
        raise ConfigError(
            f"配置含未知键（schema={schema.__name__}，请检查拼写）: {', '.join(sorted(unknown))}"
        )

    kwargs = {name: _convert_value(f, merged[name], root) for name, f in schema_fields.items()}
    return schema(**kwargs)


def _resolve_schema(name: str, _seen: set[str] | None = None) -> type:
    """解析配置的 dataclass schema。

    优先用 PRESET_MAP；否则沿 extends 链向上查找第一个已知预设。
    支持形如 "ablation/wo_dual_threshold" 的子目录名。
    """
    if _seen is None:
        _seen = set()
    if name in _seen:
        raise ConfigError(f"extends 链出现循环: {name!r}")
    _seen.add(name)
    if name in PRESET_MAP:
        return PRESET_MAP[name]
    path = _configs_file(name)
    if not path.exists():
        raise ConfigError(f"未知配置预设: {name!r}，可选: {', '.join(sorted(PRESET_MAP))}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    extends = data.get("extends")
    if extends is None:
        raise ConfigError(f"配置 {name!r} 缺少 extends，无法推断 schema")
    names = [extends] if isinstance(extends, str) else list(extends)
    for parent in names:
        if not isinstance(parent, str):
            raise ConfigError(f"extends 项必须是字符串: {parent!r}")
        if parent in PRESET_MAP:
            return PRESET_MAP[parent]
        return _resolve_schema(parent, _seen)
    raise ConfigError(f"配置 {name!r} 的 extends 链未指向任何已知预设")


def load_config(name: str, timestamp: str | None = None) -> Any:
    """按预设名（或子目录名，如 "ablation/xxx"）加载配置。

    返回类型标注为 Any：schema 由预设名在运行期动态解析（PRESET_MAP），
    静态分析无法推断具体配置类，调用方按预设对应的 dataclass 访问字段。
    """
    schema = _resolve_schema(name)
    merged = _load_yaml_file(_configs_file(name), set())
    root = project_root()
    return _validate_and_build(schema, merged, root, timestamp)


_DEFAULT_CACHE: dict[str, Any] = {}


def get_default(name: str) -> Any:
    """返回指定预设的默认配置实例（进程内缓存）。

    供库代码读取"回退默认值"使用（如门控构造参数为 None 时的默认超参），
    避免在每个调用点重复加载 YAML。入口点应优先用 load_config 显式加载。
    """
    if name not in _DEFAULT_CACHE:
        _DEFAULT_CACHE[name] = load_config(name)
    return _DEFAULT_CACHE[name]


def load_config_path(path: str | Path, timestamp: str | None = None) -> Any:
    """加载任意 YAML 文件路径（支持 extends 引用 configs/ 下的预设）。"""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    # 推断 schema：优先按文件名匹配预设，否则要求文件内显式声明 _schema
    stem = p.stem
    schema = PRESET_MAP.get(stem)
    merged = _load_yaml_file(p, set())
    if schema is None:
        schema_name = merged.pop("_schema", None)
        if schema_name not in PRESET_MAP:
            raise ConfigError(
                f"无法推断 schema（文件={p.name}）：文件名不匹配任何预设，"
                f"且未声明有效的 _schema 键。可选: {', '.join(sorted(PRESET_MAP))}"
            )
        schema = PRESET_MAP[schema_name]
    root = project_root()
    if timestamp is not None:
        merged["TIMESTAMP"] = timestamp
    return _validate_and_build(schema, merged, root)
