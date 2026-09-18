"""
@author: landfallbox
@date: 2026/02/03 星期一
@description: lstm 训练脚本
"""

from et_prl.config.loader import load_config
from et_prl.data import DatasetLoader
from et_prl.evaluation import LSTMEvaluator
from et_prl.models import LSTM
from et_prl.training import LSTMTrainer
from et_prl.utils import create_experiment_context, create_loss_fn, create_optimizer


def train_lstm():
    """LSTM训练主函数"""
    # 1. 初始化配置
    config = load_config("lstm")

    experiment_dir = config.get_train_experiment_dir()
    context = create_experiment_context(
        experiment_dir=experiment_dir,
        config_dict=config.to_dict(),
        with_checkpoint_manager=True,
        checkpoint_dir_name=config.CHECKPOINT_DIR_NAME,
        log_filename=config.RUN_LOG_FILENAME,
        results_dir=config.get_run_results_dir(),
        tb_dir=config.get_run_tb_dir(),
    )
    logger = context.logger
    metrics_recorder = context.metrics_recorder
    checkpoint_manager = context.checkpoint_manager

    logger.info("配置初始化完成")
    logger.info(f"实验名称: {config.EXPERIMENT_NAME}")
    logger.info(f"设备: {config.DEVICE}")

    # 2. 加载数据
    logger.info("加载数据...")
    dataset_loader = DatasetLoader(config)
    train_loader, val_loader, _ = dataset_loader.load_data(
        load_normalizer=True,
        reshape_for_rnn=True
    )
    logger.info(f"数据加载完成，批大小: {config.BATCH_SIZE}")

    # 3. 创建模型
    logger.info("初始化LSTM模型...")
    model = LSTM(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE,
        batch_first=config.BATCH_FIRST,
        dropout=config.DROPOUT
    )
    logger.info("模型架构:")
    logger.info(f"输入特征维度: {config.INPUT_SIZE}")
    logger.info(f"隐藏层大小: {config.HIDDEN_SIZES}")
    logger.info(f"输出维度: {config.OUTPUT_SIZE}")

    # 4. 创建优化器和损失函数
    logger.info("初始化优化器和损失函数...")
    optimizer = create_optimizer(model, config.OPTIMIZER, config.LEARNING_RATE)
    loss_fn = create_loss_fn(config.LOSS_FUNCTION)
    logger.info(f"优化器: {config.OPTIMIZER}")
    logger.info(f"学习率: {config.LEARNING_RATE}")
    logger.info(f"损失函数: {config.LOSS_FUNCTION}")

    # 5. 创建评估器
    logger.info("初始化评估器...")
    evaluator = LSTMEvaluator(model, loss_fn, config.DEVICE)

    # 6. 创建训练器
    logger.info("初始化训练器...")
    trainer = LSTMTrainer(model, optimizer, loss_fn, config.DEVICE)

    # 7. 执行训练
    logger.info("开始训练...")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        epochs=config.EPOCHS,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        config=config.to_dict(),
        early_stop_patience=config.EARLY_STOP_PATIENCE
    )
    logger.info("训练完成！")

    # 获取最优epoch信息
    best_epoch = history["best_epoch"]
    best_val_loss = history["best_val_loss"]
    stopped_epoch = history["stopped_epoch"]
    logger.info(f"最优验证loss: {best_val_loss:.4f} (Epoch {best_epoch + 1})")
    if config.EARLY_STOP_PATIENCE is not None:
        logger.info(f"实际训练停止epoch: {stopped_epoch + 1} (总配置epoch: {config.EPOCHS})")

    # 8. 保存训练历史
    logger.info("保存训练历史...")

    # 保存摘要信息到 metrics.json（轻量级）
    metrics_recorder.save_metrics({
        "summary": {
            "total_epochs": len(history["train_loss"]),
            "stopped_epoch": stopped_epoch + 1,
            "best_epoch": best_epoch + 1,
            "best_val_loss": best_val_loss,
            "early_stop_triggered": stopped_epoch < config.EPOCHS - 1 if config.EARLY_STOP_PATIENCE else False
        }
    })

    # 保存详细的训练历史到 CSV（便于分析和绘图）
    metrics_recorder.save_training_history(history)
    logger.info("训练历史已保存")

    logger.info("训练流程完成！")


def main() -> None:
    """CLI 入口函数"""
    train_lstm()
