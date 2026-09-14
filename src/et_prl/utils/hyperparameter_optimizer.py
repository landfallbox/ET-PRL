"""
@Author      : landfallbox
@Date        : 2026/02/04 星期二
@Description : 通用贝叶斯超参优化器
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler, TPESampler


class HyperparameterSpace:
    """超参搜索空间定义类"""

    def __init__(self):
        """初始化搜索空间"""
        self.params: Dict[str, Dict[str, Any]] = {}

    def add_int(self, name: str, low: int, high: int) -> "HyperparameterSpace":
        """
        添加整数类型超参

        参数：
            name: 超参名称
            low: 最小值（包含）
            high: 最大值（包含）
        """
        self.params[name] = {"type": "int", "low": low, "high": high}
        return self

    def add_float(
        self, name: str, low: float, high: float, log: bool = False
    ) -> "HyperparameterSpace":
        """
        添加浮点数类型超参

        参数：
            name: 超参名称
            low: 最小值
            high: 最大值
            log: 是否使用对数尺度
        """
        self.params[name] = {"type": "float", "low": low, "high": high, "log": log}
        return self

    def add_categorical(self, name: str, choices: list) -> "HyperparameterSpace":
        """
        添加分类类型超参

        参数：
            name: 超参名称
            choices: 可选值列表
        """
        self.params[name] = {"type": "categorical", "choices": choices}
        return self

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """获取搜索空间字典"""
        return self.params


class BayesianOptimizer:
    """贝叶斯超参优化器"""

    DEFAULT_ROUND_DECIMALS = 6

    def __init__(
        self,
        space: HyperparameterSpace,
        output_dir: Path,
        sampler: str = "tpe",
        seed: Optional[int] = None,
    ):
        """
        初始化贝叶斯优化器

        参数：
            space: 超参搜索空间
            output_dir: 结果保存目录
            sampler: 采样器类型（tpe, random）
            seed: 随机种子
        """
        self.space = space
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sampler = sampler
        self.seed = seed
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None
        self.study: Optional[optuna.Study] = None
        self.optimization_history: list = []

    def _create_sampler(self):
        """创建采样器"""
        if self.sampler == "tpe":
            return TPESampler(seed=self.seed)
        elif self.sampler == "random":
            return RandomSampler(seed=self.seed)
        else:
            raise ValueError(f"不支持的采样器类型: {self.sampler}")

    def _objective_wrapper(self, objective_fn: Callable, **objective_kwargs) -> Callable:
        """
        包装目标函数，使其适配 Optuna 接口

        参数：
            objective_fn: 用户定义的目标函数，签名为 (trial, params, **kwargs) -> float
            objective_kwargs: 传递给目标函数的额外关键字参数

        返回：
            Optuna 兼容的目标函数
        """

        def optuna_objective(trial: optuna.Trial) -> float:
            # 从 trial 中建议超参
            params = {}
            for param_name, param_config in self.space.to_dict().items():
                param_type = param_config["type"]
                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False),
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

            # 调用用户目标函数
            value = objective_fn(trial, params, **objective_kwargs)
            return value

        return optuna_objective

    def optimize(
        self,
        objective_fn: Callable,
        n_trials: int = 100,
        n_jobs: int = 1,
        **objective_kwargs,
    ) -> Dict[str, Any]:
        """
        执行贝叶斯优化

        参数：
            objective_fn: 目标函数，签名为 (trial, params, **kwargs) -> float
                         应返回单个数值（优化目标，通常为验证损失）
            n_trials: 优化轮数
            n_jobs: 并行任务数（1 表示串行）
            objective_kwargs: 传递给目标函数的额外关键字参数

        返回：
            包含最优超参和最优值的字典
        """
        optuna_objective = self._objective_wrapper(objective_fn, **objective_kwargs)

        # 创建 Study
        sampler = self._create_sampler()
        pruner = MedianPruner()
        self.study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        # 执行优化
        self.study.optimize(optuna_objective, n_trials=n_trials, n_jobs=n_jobs)

        # 记录最优结果
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        # 保存优化历史
        self._save_results()

        if self.study is None:
            raise RuntimeError("优化未成功创建 study，无法返回结果")

        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
        }

    def _round_param_value(self, name: str, value: Any) -> Any:
        """仅对连续超参进行四舍五入，保持搜索过程连续"""
        if name in self.space.to_dict():
            param_config = self.space.to_dict()[name]
            if param_config.get("type") == "float" and isinstance(value, float):
                return round(value, self.DEFAULT_ROUND_DECIMALS)
        return value

    def _round_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """对超参字典中的连续值做四舍五入"""
        if params is None:
            return {}
        return {name: self._round_param_value(name, value) for name, value in params.items()}

    def _save_results(self):
        """保存优化结果到文件"""
        if self.study is None:
            raise RuntimeError("study 未初始化，无法保存优化结果")

        study = self.study
        results = {
            "best_params": self._round_params(self.best_params),
            "best_params_raw": self.best_params,
            "best_params_rounded": self._round_params(self.best_params),
            "best_value": self.best_value,
            "n_trials": len(study.trials),
            "trials": [],
        }

        # 记录每一轮的结果
        for trial in study.trials:
            trial_record = {
                "trial_number": trial.number,
                "params": self._round_params(trial.params),
                "params_raw": trial.params,
                "params_rounded": self._round_params(trial.params),
                "value": trial.value,
                "state": trial.state.name,
            }
            results["trials"].append(trial_record)

        # 保存为 JSON
        output_file = self.output_dir / "optimization_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def get_optimization_history(self) -> list:
        """
        获取优化历史

        返回：
            包含每一轮结果的列表
        """
        if self.study is None:
            return []

        history = []
        for trial in self.study.trials:
            history.append(
                {
                    "trial": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": trial.state.name,
                }
            )
        return history

    def get_best_params(self) -> Dict[str, Any]:
        """获取最优超参"""
        if self.best_params is None:
            raise RuntimeError("尚未完成优化，best_params 不可用")
        return self.best_params

    def get_best_value(self) -> float:
        """获取最优值（验证损失）"""
        if self.best_value is None:
            raise RuntimeError("尚未完成优化，best_value 不可用")
        return self.best_value
