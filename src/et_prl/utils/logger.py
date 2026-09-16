"""
@Author      : landfallbox
@Date        : 2026/02/03 星期一
@Description : 日志记录器（通用可复用实现）
"""

import logging
import uuid
from pathlib import Path


class Logger:
    """
    日志记录器

    职责：
    - 文本日志输出到控制台和文件
    - 提供不同级别的日志记录方法（info, debug, warning, error）
    """

    def __init__(self, experiment_dir: Path, log_filename: str = "run.log"):
        """
        初始化日志器

        参数：
            experiment_dir: 实验目录
            log_filename: 日志文件名（可选，默认为 'run.log'）
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.log_filename = log_filename

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """配置日志器。

        使用 UUID 命名 logger，避免同名实验目录共享 logger 导致日志串流。
        """
        logger = logging.getLogger(f"et_prl.{uuid.uuid4().hex[:8]}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            log_file = self.experiment_dir / self.log_filename
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    def info(self, msg: str):
        """INFO 级别日志"""
        self.logger.info(msg, stacklevel=2)

    def debug(self, msg: str):
        """DEBUG 级别日志"""
        self.logger.debug(msg, stacklevel=2)

    def warning(self, msg: str):
        """WARNING 级别日志"""
        self.logger.warning(msg, stacklevel=2)

    def error(self, msg: str):
        """ERROR 级别日志"""
        self.logger.error(msg, stacklevel=2)

    def close(self) -> None:
        """刷新并释放日志文件句柄。"""
        for handler in list(self.logger.handlers):
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass
            self.logger.removeHandler(handler)
