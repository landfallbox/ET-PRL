"""
@Author      : landfallbox
@Date        : 2026/02/06 星期四
@Description : 可视化工具类
"""

import warnings
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


class Visualizer:
    """
    可视化工具类

    职责：
    - 提供各种数据可视化方法
    - 生成预测对比图、误差分析图等
    - 统一图表样式和输出格式
    """

    _font_configured = False
    _has_chinese_font = False

    @staticmethod
    def _setup_chinese_font():
        """
        设置中文字体支持，自动检测可用字体

        优先级：
        1. SimHei（黑体）
        2. Microsoft YaHei（微软雅黑）
        3. STSong（华文宋体，Mac）
        4. 系统默认中文字体
        5. 如果没有中文字体，使用 Arial 并警告
        """
        if Visualizer._font_configured:
            return

        # 获取系统所有可用字体
        available_fonts = set([f.name for f in fm.fontManager.ttflist])

        # 按优先级尝试中文字体
        chinese_fonts = [
            "SimHei",  # Windows 黑体
            "Microsoft YaHei",  # Windows 微软雅黑
            "STSong",  # Mac 华文宋体
            "STHeiti",  # Mac 黑体
            "PingFang SC",  # Mac 苹方
            "Noto Sans CJK SC",  # Linux Noto
            "WenQuanYi Micro Hei",  # Linux 文泉驿微米黑
        ]

        # 查找第一个可用的中文字体
        selected_font = None
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                Visualizer._has_chinese_font = True
                break

        # 设置字体
        if selected_font:
            plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans", "Arial"]
        else:
            # 没有中文字体，使用默认字体
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
            # 只在第一次配置时警告
            warnings.warn(
                "未找到中文字体，图表中的中文可能显示为方框。"
                "建议安装 SimHei 或 Microsoft YaHei 字体以获得更好的显示效果。",
                UserWarning,
            )

        # 解决负号显示问题
        plt.rcParams["axes.unicode_minus"] = False

        Visualizer._font_configured = True

    @staticmethod
    def plot_prediction_comparison(
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Path,
        max_samples: int = 500,
        title: str = "模型预测结果对比",
        xlabel: str = "样本索引",
        ylabel: str = "目标值",
        figsize: Tuple[int, int] = (15, 6),
        dpi: int = 300,
    ) -> None:
        """
        绘制预测值与真实值的对比折线图

        参数：
            predictions: 预测值数组
            targets: 真实值数组
            save_path: 保存路径
            max_samples: 最大显示样本数，避免图表过于密集
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            figsize: 图表尺寸
            dpi: 输出分辨率
        """
        num_samples = min(len(predictions), max_samples)
        indices = np.linspace(0, len(predictions) - 1, num_samples, dtype=int)

        pred_sample = predictions[indices]
        target_sample = targets[indices]

        Visualizer._setup_chinese_font()

        plt.figure(figsize=figsize)

        plt.plot(
            range(num_samples),
            target_sample,
            label="真实值",
            linewidth=1.5,
            alpha=0.8,
            color="#2E86AB",
        )
        plt.plot(
            range(num_samples),
            pred_sample,
            label="预测值",
            linewidth=1.5,
            alpha=0.8,
            color="#A23B72",
        )

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_error_distribution(
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Path,
        bins: int = 50,
        title: str = "预测误差分布",
        xlabel: str = "误差值",
        ylabel: str = "频数",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 300,
    ) -> None:
        """
        绘制预测误差的分布直方图

        参数：
            predictions: 预测值数组
            targets: 真实值数组
            save_path: 保存路径
            bins: 直方图柱数
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            figsize: 图表尺寸
            dpi: 输出分辨率
        """
        errors = predictions - targets

        Visualizer._setup_chinese_font()

        plt.figure(figsize=figsize)

        plt.hist(errors, bins=bins, color="#6A4C93", alpha=0.7, edgecolor="black")

        mean_error = np.mean(errors)
        std_error = np.std(errors)
        plt.axvline(
            mean_error, color="red", linestyle="--", linewidth=2, label=f"均值: {mean_error:.4f}"
        )
        plt.axvline(
            mean_error + std_error,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"±标准差: {std_error:.4f}",
        )
        plt.axvline(mean_error - std_error, color="orange", linestyle="--", linewidth=1.5)

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, linestyle="--", axis="y")
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_scatter_comparison(
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Path,
        title: str = "预测值vs真实值散点图",
        xlabel: str = "真实值",
        ylabel: str = "预测值",
        figsize: Tuple[int, int] = (8, 8),
        dpi: int = 300,
    ) -> None:
        """
        绘制预测值与真实值的散点图

        参数：
            predictions: 预测值数组
            targets: 真实值数组
            save_path: 保存路径
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            figsize: 图表尺寸
            dpi: 输出分辨率
        """
        Visualizer._setup_chinese_font()

        plt.figure(figsize=figsize)

        plt.scatter(targets, predictions, alpha=0.5, s=20, color="#1B998B")

        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="理想预测线")

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.axis("equal")
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_training_history(
        train_losses: np.ndarray,
        val_losses: np.ndarray,
        save_path: Path,
        title: str = "训练过程损失曲线",
        xlabel: str = "Epoch",
        ylabel: str = "Loss",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 300,
    ) -> None:
        """
        绘制训练过程的损失曲线

        参数：
            train_losses: 训练损失数组
            val_losses: 验证损失数组
            save_path: 保存路径
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            figsize: 图表尺寸
            dpi: 输出分辨率
        """
        Visualizer._setup_chinese_font()

        plt.figure(figsize=figsize)

        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, label="训练损失", linewidth=2, alpha=0.8, color="#FF6B6B")
        plt.plot(epochs, val_losses, label="验证损失", linewidth=2, alpha=0.8, color="#4ECDC4")

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
