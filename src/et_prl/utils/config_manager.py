"""
@Author      : landfallbox
@Date        : 2026/02/04 星期二
@Description : 配置管理器（通用可复用实现）
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    """
    配置管理器

    职责：
    - 保存配置到 YAML 文件
    - 加载配置从 YAML 文件
    """

    def __init__(self, experiment_dir: Path, config_filename: str = "config.yaml"):
        """
        初始化配置管理器

        参数：
            experiment_dir: 实验目录
            config_filename: 配置文件名（默认为 'config.yaml'）
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.config_filename = config_filename

    def save_config(self, config: Dict[str, Any], filename: Optional[str] = None):
        """
        保存配置到 YAML 文件

        参数：
            config: 配置字典
            filename: 配置文件名（可选，如果不提供则使用初始化时的文件名）
        """
        if filename is None:
            filename = self.config_filename
        config_file = self.experiment_dir / filename
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)

    def load_config(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        从 YAML 文件加载配置

        参数：
            filename: 配置文件名（可选，如果不提供则使用初始化时的文件名）

        返回：
            配置字典
        """
        if filename is None:
            filename = self.config_filename
        config_file = self.experiment_dir / filename
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
