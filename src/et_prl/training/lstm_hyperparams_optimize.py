"""
@author: landfallbox
@date: 2026/02/07 星期六
@description: LSTM 超参优化脚本
"""
from dataclasses import replace

from et_prl.config.loader import load_config
from et_prl.data import DatasetLoader
from et_prl.evaluation import LSTMEvaluator
from et_prl.models import LSTM
from et_prl.training import LSTMTrainer
from et_prl.utils import HyperparameterSpace, BayesianOptimizer, Logger, create_loss_fn, create_optimizer


def adjust_hyperparams_by_correlation(params: dict) -> dict:
    """
    根据参数相关性关系调整超参

    调整规则：
    1. 大batch size需要配合更高的学习率（梯度累积效应）
       - batch_size >= 96: lr * 1.5
       - batch_size >= 64: lr * 1.2
       - batch_size < 24: lr * 0.8

    2. 深层网络（隐藏层大小大）应该提高dropout以防止过拟合
       - 隐藏层总大小 > 200: dropout * 1.5
       - 隐藏层总大小 > 150: dropout * 1.3

    参数：
        params: 原始超参字典

    返回：
        调整后的超参字典
    """
    adjusted_params = params.copy()

    # 规则1：大batch size搭配更高学习率
    batch_size = adjusted_params.get("batch_size", 32)
    base_lr = adjusted_params.get("learning_rate", 0.001)

    if batch_size >= 96:
        adjusted_params["learning_rate"] = min(base_lr * 1.5, 5e-3)
    elif batch_size >= 64:
        adjusted_params["learning_rate"] = min(base_lr * 1.2, 5e-3)
    elif batch_size < 24:
        adjusted_params["learning_rate"] = max(base_lr * 0.8, 1e-4)

    # 规则2：深层网络提高dropout
    if "hidden_size_1" in adjusted_params and "hidden_size_2" in adjusted_params:
        total_hidden_size = adjusted_params["hidden_size_1"] + adjusted_params["hidden_size_2"]
        base_dropout = adjusted_params.get("dropout", 0.0)

        if total_hidden_size > 200:
            adjusted_params["dropout"] = min(base_dropout * 1.5, 0.35)
        elif total_hidden_size > 150:
            adjusted_params["dropout"] = min(base_dropout * 1.3, 0.3)

    return adjusted_params


def create_objective_fn(train_loader, val_loader, base_config, logger):
    """
    创建目标函数工厂

    参数：
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        base_config: 基础配置实例（每个 trial 基于它派生，避免重复加载 YAML）
        logger: 日志记录器

    返回：
        目标函数 (trial, params) -> float
    """
    def objective(trial, params):
        """
        优化目标函数，默认最小化验证损失

        参数：
            trial: Optuna trial 对象
            params: 超参字典

        返回：
            验证损失（单个浮点数）
        """
        try:
            config = base_config

            adjusted_params = adjust_hyperparams_by_correlation(params)

            config = replace(
                config,
                LEARNING_RATE=adjusted_params.get("learning_rate", config.LEARNING_RATE),
                BATCH_SIZE=adjusted_params.get("batch_size", config.BATCH_SIZE),
                EPOCHS=adjusted_params.get("epochs", config.EPOCHS),
                DROPOUT=adjusted_params.get("dropout", config.DROPOUT),
                OPTIMIZER=adjusted_params.get("optimizer", config.OPTIMIZER),
                HIDDEN_SIZES=[
                    adjusted_params.get("hidden_size_1", config.HIDDEN_SIZES[0]),
                    adjusted_params.get("hidden_size_2", config.HIDDEN_SIZES[1]),
                ],
            )

            logger.info(
                f"Trial {trial.number}: 原始超参 - LR={params.get('learning_rate', 0.001):.2e}, "
                f"BS={params.get('batch_size', 32)}, Dropout={params.get('dropout', 0.0):.3f}"
            )
            logger.info(
                f"Trial {trial.number}: 调整后 - LR={config.LEARNING_RATE:.2e}, "
                f"BS={config.BATCH_SIZE}, Dropout={config.DROPOUT:.3f}, "
                f"H1={config.HIDDEN_SIZES[0]}, H2={config.HIDDEN_SIZES[1]}"
            )

            model = LSTM(
                input_size=config.INPUT_SIZE,
                hidden_sizes=config.HIDDEN_SIZES,
                output_size=config.OUTPUT_SIZE,
                batch_first=config.BATCH_FIRST,
                dropout=config.DROPOUT
            )

            optimizer = create_optimizer(model, config.OPTIMIZER, config.LEARNING_RATE)
            loss_fn = create_loss_fn(config.LOSS_FUNCTION)

            evaluator = LSTMEvaluator(model, loss_fn, config.DEVICE)
            trainer = LSTMTrainer(model, optimizer, loss_fn, config.DEVICE)

            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                evaluator=evaluator,
                epochs=config.EPOCHS,
                logger=None,
                checkpoint_manager=None,
                config=config.to_dict(),
                early_stop_patience=config.EARLY_STOP_PATIENCE
            )

            best_val_loss = history["best_val_loss"]
            logger.info(f"Trial {trial.number}: 最优验证损失 = {best_val_loss:.6f}")

            return best_val_loss

        except Exception as e:
            logger.error(f"Trial {trial.number} 执行失败: {str(e)}")
            return float("inf")

    return objective


def optimize_lstm_hyperparameters(n_trials: int = 80):
    """超参优化主函数

    参数：
        n_trials: 贝叶斯优化试验次数
    """
    config = load_config("lstm")
    output_dir = config.get_optimization_dir() / config.TIMESTAMP
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(output_dir)
    logger.info("开始LSTM超参贝叶斯优化")

    logger.info("加载数据...")
    dataset_loader = DatasetLoader(config)
    train_loader, val_loader, _ = dataset_loader.load_data(
        load_normalizer=True,
        reshape_for_rnn=True
    )
    logger.info(f"数据加载完成，批大小: {config.BATCH_SIZE}")

    logger.info("定义超参搜索空间...")
    space = HyperparameterSpace()
    space.add_float("learning_rate", 1e-4, 5e-3, log=True) \
         .add_int("batch_size", 16, 128) \
         .add_int("epochs", 30, 150) \
         .add_float("dropout", 0.0, 0.3) \
         .add_int("hidden_size_1", 32, 128) \
         .add_int("hidden_size_2", 16, 64) \
         .add_categorical("optimizer", ["adam", "sgd"])

    logger.info("搜索空间已定义:")
    for param_name, param_config in space.to_dict().items():
        logger.info(f"{param_name}: {param_config}")

    logger.info("初始化贝叶斯优化器...")
    optimizer = BayesianOptimizer(
        space=space,
        output_dir=output_dir,
        results_dir=output_dir / "results",
        sampler="tpe",
        seed=42
    )

    objective_fn = create_objective_fn(
        train_loader=train_loader,
        val_loader=val_loader,
        base_config=config,
        logger=logger
    )

    logger.info("开始优化...")
    result = optimizer.optimize(
        objective_fn=objective_fn,
        n_trials=int(n_trials)
    )

    logger.info("优化完成！")
    logger.info(f"总试验数: {result['n_trials']}")
    logger.info(f"最优验证损失: {result['best_value']:.6f}")
    logger.info("最优超参:")
    for param_name, param_value in result["best_params"].items():
        logger.info(f"{param_name}: {param_value}")

    logger.info(f"优化结果已保存到: {output_dir / 'results' / 'optimization_results.json'}")

    return result


def main() -> None:
    """CLI 入口函数"""
    optimize_lstm_hyperparameters()

