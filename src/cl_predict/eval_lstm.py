"""
@Author      : landfallbox
@Date        : 2026/02/04 星期二
@Description : LSTM 模型评估脚本
"""
import argparse
import json
from pathlib import Path

from config.lstm_config import LSTMConfig
from ml_toolkit.data_processing import DatasetLoader
from ml_toolkit.evaluation import LSTMEvaluator
from ml_toolkit.evaluation import calculate_mae, calculate_rmse, calculate_mape, calculate_r2_score
from ml_toolkit.models import LSTM
from ml_toolkit.utils import CheckpointManager, ConfigManager, Logger, create_loss_fn, Visualizer


def eval_lstm(train_experiment_dir: Path = None):
    """
    LSTM 模型评估主函数

    参数：
        train_experiment_dir: 指定训练实验目录，如果为 None 则自动加载最新训练实验
    """
    # 1. 初始化配置并创建评估专用实验目录
    config = LSTMConfig()
    eval_experiment_dir = config.get_eval_experiment_dir()
    logger = Logger(eval_experiment_dir, log_filename=config.EVALUATION_LOG_FILENAME)
    config_manager = ConfigManager(eval_experiment_dir)

    logger.info("=" * 50)
    logger.info("开始模型评估")
    logger.info(f"评估实验目录: {eval_experiment_dir}")

    # 2. 查找或指定训练实验目录
    if train_experiment_dir is None:
        logger.info("未指定训练实验目录，正在查找最新训练实验...")
        train_experiment_dir = CheckpointManager.find_latest_experiment("lstm", mode="train")

        if train_experiment_dir is None:
            logger.error("未找到任何 LSTM 训练实验目录")
            return

        logger.info(f"找到最新训练实验: {train_experiment_dir}")
    else:
        train_experiment_dir = Path(train_experiment_dir)
        if not train_experiment_dir.exists():
            logger.error(f"指定的训练实验目录不存在: {train_experiment_dir}")
            return
        logger.info(f"使用指定的训练实验目录: {train_experiment_dir}")

    # 3. 加载训练实验配置
    logger.info("加载训练实验配置...")
    train_config_manager = ConfigManager(train_experiment_dir)
    config_dict = train_config_manager.load_config()

    # 保存训练配置到评估目录（便于追溯）
    config_manager.save_config(config_dict)

    # 重建配置对象（用于数据加载）
    config = LSTMConfig()
    logger.info("配置加载完成")
    logger.info(f"实验名称: {config.EXPERIMENT_NAME}")
    logger.info(f"设备: {config.DEVICE}")

    # 4. 加载测试数据
    logger.info("加载测试数据...")
    dataset_loader = DatasetLoader(config)
    _, _, test_loader = dataset_loader.load_data(
        load_normalizer=True,
        reshape_for_rnn=True
    )
    logger.info("测试数据加载完成")

    # 5. 创建模型
    logger.info("初始化LSTM模型...")
    model = LSTM(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE,
        batch_first=config.BATCH_FIRST,
        dropout=config.DROPOUT
    )
    logger.info("模型架构:")
    logger.info(f"  输入特征维度: {config.INPUT_SIZE}")
    logger.info(f"  隐藏层大小: {config.HIDDEN_SIZES}")
    logger.info(f"  输出维度: {config.OUTPUT_SIZE}")

    # 6. 加载最优模型权重
    logger.info("加载最优模型权重...")
    checkpoint = CheckpointManager.load_best_model(
        train_experiment_dir,
        model,
        map_location=config.DEVICE
    )

    best_epoch = checkpoint.get('epoch', 'Unknown')
    best_metrics = checkpoint.get('metrics', {})
    logger.info(f"最优模型来自 Epoch {best_epoch + 1}")
    if best_metrics:
        logger.info("训练时的验证指标:")
        for key, value in best_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    # 7. 创建损失函数和评估器
    logger.info("初始化评估器...")
    loss_fn = create_loss_fn(config.LOSS_FUNCTION)
    metrics = {
        "mae": calculate_mae,
        "rmse": calculate_rmse,
        "mape": calculate_mape,
        "r2": calculate_r2_score
    }
    evaluator = LSTMEvaluator(model, loss_fn, config.DEVICE, metrics=metrics)

    # 8. 在测试集上进行评估
    logger.info("在测试集上进行评估...")
    test_metrics = evaluator.test(test_loader)

    logger.info("测试集评估结果:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # 9. 收集预测值和真实值用于可视化
    logger.info("收集预测值和真实值...")
    predictions, targets = evaluator.predict(test_loader)

    predictions_np = predictions.squeeze().numpy()
    targets_np = targets.squeeze().numpy()

    logger.info(f"预测值形状: {predictions_np.shape}")
    logger.info(f"真实值形状: {targets_np.shape}")

    # 10. 生成多种可视化图表
    logger.info("生成可视化图表...")

    # 预测对比折线图
    plot_save_path = eval_experiment_dir / config.PREDICTION_COMPARISON_PLOT_FILENAME
    Visualizer.plot_prediction_comparison(
        predictions_np, targets_np, plot_save_path,
        max_samples=config.MAX_PLOT_SAMPLES,
        title="LSTM模型预测结果对比",
        figsize=config.PREDICTION_COMPARISON_FIGSIZE,
        dpi=config.PLOT_DPI
    )
    logger.info(f"预测对比图已保存: {plot_save_path}")

    # 误差分布直方图
    error_dist_path = eval_experiment_dir / config.ERROR_DISTRIBUTION_PLOT_FILENAME
    Visualizer.plot_error_distribution(
        predictions_np, targets_np, error_dist_path,
        bins=config.ERROR_HIST_BINS,
        figsize=config.ERROR_DISTRIBUTION_FIGSIZE,
        dpi=config.PLOT_DPI
    )
    logger.info(f"误差分布图已保存: {error_dist_path}")

    # 预测散点图
    scatter_path = eval_experiment_dir / config.PREDICTION_SCATTER_PLOT_FILENAME
    Visualizer.plot_scatter_comparison(
        predictions_np, targets_np, scatter_path,
        figsize=config.PREDICTION_SCATTER_FIGSIZE,
        dpi=config.PLOT_DPI
    )
    logger.info(f"预测散点图已保存: {scatter_path}")

    # 11. 保存测试结果到评估实验目录
    logger.info("保存测试结果...")

    # 保存评估结果到评估实验目录
    metrics_file = eval_experiment_dir / config.EVALUATION_METRICS_FILENAME
    eval_results = {
        "train_experiment_dir": str(train_experiment_dir),
        "test_metrics": test_metrics,
        "best_epoch_from_train": best_epoch + 1 if best_epoch != 'Unknown' else 'Unknown',
        "best_val_metrics_from_train": best_metrics
    }

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    logger.info(f"测试结果已保存: {metrics_file}")
    logger.info("评估完成！")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="LSTM 模型评估脚本")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="指定训练实验目录路径，如果不指定则自动加载最新训练实验"
    )

    args = parser.parse_args()

    eval_lstm(train_experiment_dir=args.experiment_dir)


if __name__ == "__main__":
    main()
