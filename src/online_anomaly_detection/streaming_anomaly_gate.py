"""
在线异常检测网关
"""
import pickle
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass
from pathlib import Path

from config.online_anomaly_detection_config import OnlineAnomalyDetectionConfig
from ml_toolkit.anomaly_detection import MultiScaleDistributionTracker
from ml_toolkit.anomaly_detection import StreamingIsolationDepth
from ml_toolkit.anomaly_detection import StreamingStats
from ml_toolkit.anomaly_detection import StreamingThresholdOptimizer


@dataclass
class GateDecision:
    """网关决策结果"""

    gate_signal: int  # 0=正常, 1=异常(触发DQN)
    anomaly_score: float  # 0-1的异常分数
    confidence: float  # 0-1的置信度
    adaptive_threshold: float  # 当前自适应阈值
    timestamp: Optional[float] = None  # 样本时间戳


class StreamingAnomalyGate:
    """
    完全在线学习的异常检测门控网络

    核心特性：
    1. 流式异常检测 - 不依赖离线模型，实时适应数据分布
    2. 双层自适应阈值 - 本地快速响应 + 全局长期稳定
    3. 多尺度分布追踪 - 同时捕捉短期和长期的数据漂移
    4. 零外部依赖 - 完全独立运行，不需要后续的分类模型
    """

    def __init__(
        self,
        feature_dim: int,
        local_window_size: Optional[int] = None,
        global_ema_decay: Optional[float] = None,
        reference_samples: Optional[int] = None,
        contamination: Optional[float] = None,
        alpha_local_weight: Optional[float] = None,
        threshold_bias: Optional[float] = None,
        threshold_quantile: Optional[float] = None,
        threshold_mad_scale: Optional[float] = None,
        threshold_local_update_rate: Optional[float] = None,
        threshold_quantile_weight: Optional[float] = None,
        score_short_weight: Optional[float] = None,
        score_medium_weight: Optional[float] = None,
        score_long_weight: Optional[float] = None,
        trigger_hysteresis_margin: Optional[float] = None,
        min_trigger_interval_steps: Optional[int] = None,
    ):
        """
        初始化在线异常检测门控

        参数：
            feature_dim: 特征维度（输入特征数）
            local_window_size: 本地阈值窗口大小
            global_ema_decay: 全局阈值EMA衰减率（越小越稳定）
            reference_samples: 流式IF参考集大小
            contamination: 异常比例先验
            alpha_local_weight: 本地阈值权重（0-1，越大越快反应漂移）
        """
        self.feature_dim = feature_dim
        self.sample_count = 0

        if local_window_size is None:
            local_window_size = OnlineAnomalyDetectionConfig.GATE_LOCAL_WINDOW_SIZE
        if global_ema_decay is None:
            global_ema_decay = OnlineAnomalyDetectionConfig.GATE_GLOBAL_EMA_DECAY
        if reference_samples is None:
            reference_samples = OnlineAnomalyDetectionConfig.GATE_REFERENCE_SAMPLES
        if contamination is None:
            contamination = OnlineAnomalyDetectionConfig.GATE_CONTAMINATION
        if alpha_local_weight is None:
            alpha_local_weight = OnlineAnomalyDetectionConfig.GATE_ALPHA_LOCAL_WEIGHT
        if threshold_bias is None:
            threshold_bias = OnlineAnomalyDetectionConfig.GATE_THRESHOLD_BIAS
        if threshold_quantile is None:
            threshold_quantile = OnlineAnomalyDetectionConfig.THRESHOLD_QUANTILE
        if threshold_mad_scale is None:
            threshold_mad_scale = OnlineAnomalyDetectionConfig.THRESHOLD_MAD_SCALE
        if threshold_local_update_rate is None:
            threshold_local_update_rate = OnlineAnomalyDetectionConfig.THRESHOLD_LOCAL_UPDATE_RATE
        if threshold_quantile_weight is None:
            threshold_quantile_weight = OnlineAnomalyDetectionConfig.THRESHOLD_QUANTILE_WEIGHT
        if score_short_weight is None:
            score_short_weight = OnlineAnomalyDetectionConfig.GATE_SCORE_SHORT_WEIGHT
        if score_medium_weight is None:
            score_medium_weight = OnlineAnomalyDetectionConfig.GATE_SCORE_MEDIUM_WEIGHT
        if score_long_weight is None:
            score_long_weight = OnlineAnomalyDetectionConfig.GATE_SCORE_LONG_WEIGHT
        if trigger_hysteresis_margin is None:
            trigger_hysteresis_margin = OnlineAnomalyDetectionConfig.GATE_TRIGGER_HYSTERESIS_MARGIN
        if min_trigger_interval_steps is None:
            min_trigger_interval_steps = OnlineAnomalyDetectionConfig.GATE_MIN_TRIGGER_INTERVAL_STEPS

        score_weight_sum = float(score_short_weight + score_medium_weight + score_long_weight)
        if score_weight_sum <= 0.0:
            raise ValueError("分数融合权重之和必须大于 0")

        self.score_short_weight = float(score_short_weight / score_weight_sum)
        self.score_medium_weight = float(score_medium_weight / score_weight_sum)
        self.score_long_weight = float(score_long_weight / score_weight_sum)
        self.threshold_bias = float(threshold_bias)
        self.trigger_hysteresis_margin = float(max(0.0, trigger_hysteresis_margin))
        self.min_trigger_interval_steps = int(max(0, min_trigger_interval_steps))

        # 1. 流式特征统计维护器
        self.feature_stats = StreamingStats(
            feature_dim=feature_dim,
            ema_decay=OnlineAnomalyDetectionConfig.STATS_EMA_DECAY,
            window_size=OnlineAnomalyDetectionConfig.STATS_WINDOW_SIZE,
        )

        # 2. 流式异常检测器（替代IsolationForest）
        self.anomaly_detector = StreamingIsolationDepth(
            n_reference_samples=reference_samples,
            update_freq=OnlineAnomalyDetectionConfig.ISOLATION_UPDATE_FREQ,
            distance_metric=OnlineAnomalyDetectionConfig.ISOLATION_DISTANCE_METRIC,
            contamination=contamination,
            decay_strategy=OnlineAnomalyDetectionConfig.ISOLATION_DECAY_STRATEGY,
        )

        # 3. 双层阈值优化器
        self.threshold_optimizer = StreamingThresholdOptimizer(
            local_window_size=local_window_size,
            global_ema_decay=global_ema_decay,
            alpha=alpha_local_weight,
            min_samples_for_optimization=OnlineAnomalyDetectionConfig.THRESHOLD_MIN_SAMPLES_FOR_OPTIMIZATION,
            quantile=threshold_quantile,
            mad_scale=threshold_mad_scale,
            local_update_rate=threshold_local_update_rate,
            quantile_weight=threshold_quantile_weight,
        )

        # 4. 多尺度分布追踪器
        self.multi_scale_tracker = MultiScaleDistributionTracker(
            feature_dim=feature_dim,
            windows=OnlineAnomalyDetectionConfig.TRACKER_WINDOWS,
            ema_decay=OnlineAnomalyDetectionConfig.TRACKER_EMA_DECAY,
        )

        # 初始化状态
        self._initialized = False
        self._last_trigger_step = -10**9

    def initialize_with_data(self, initial_data: np.ndarray) -> None:
        """
        用离线初始化数据初始化网关（可选）

        如果有历史数据，可用它来初始化统计信息
        这有助于改善冷启动性能

        参数：
            initial_data: 初始化数据，shape (n_samples, feature_dim)
        """
        assert initial_data.shape[1] == self.feature_dim

        print(f"用{len(initial_data)}个样本初始化在线异常检测网关...")

        self.feature_stats._initialize(initial_data)
        self.anomaly_detector._initialize_reference_set(initial_data)

        # 初始化阈值
        normalized_data = np.array(
            [self.feature_stats.normalize(sample) for sample in initial_data]
        )
        scores = self.anomaly_detector.score_batch(normalized_data, window="long")

        # 设置初始阈值（无先验事件率，使用分数中位数）
        initial_threshold = float(np.median(scores))
        self.threshold_optimizer.global_threshold = initial_threshold
        self.threshold_optimizer.local_threshold = initial_threshold
        self.threshold_optimizer.adaptive_threshold = initial_threshold

        self._initialized = True
        print("初始化完成")

    def predict(
        self,
        sample: np.ndarray,
        timestamp: Optional[float] = None,
        feedback_label: Optional[int] = None,
    ) -> GateDecision:
        """
        对单个样本进行异常判断（核心推理方法）

        参数：
            sample: 输入特征，shape (feature_dim,)
            timestamp: 样本时间戳（可选）
            feedback_label: 人工反馈标签（0=正常, 1=异常），用于优化（可选）

        返回：
            GateDecision: 门控决策结果
        """
        assert sample.shape[0] == self.feature_dim

        self.sample_count += 1

        # 1. 特征归一化
        normalized_sample = self.feature_stats.normalize(sample, use_global=True)

        # 2. 计算多尺度异常分数
        short_score = self.anomaly_detector.score(normalized_sample, window="short")
        medium_score = self.anomaly_detector.score(normalized_sample, window="medium")
        long_score = self.anomaly_detector.score(normalized_sample, window="long")

        # 3. 融合多尺度分数（加权平均）
        # 权重配置：短期50% (快速响应) + 中期30% (稳定) + 长期20% (趋势)
        fused_anomaly_score = (
            self.score_short_weight * short_score
            + self.score_medium_weight * medium_score
            + self.score_long_weight * long_score
        )

        # 4. 获取自适应阈值
        adaptive_threshold = self.threshold_optimizer.get_adaptive_threshold() + self.threshold_bias
        adaptive_threshold = float(np.clip(adaptive_threshold, 0.0, 1.0))

        # 5. 做二值决策（含防抖约束）
        trigger_threshold = float(np.clip(adaptive_threshold + self.trigger_hysteresis_margin, 0.0, 1.0))
        is_above_trigger_threshold = fused_anomaly_score > trigger_threshold
        enough_interval = (self.sample_count - self._last_trigger_step) > self.min_trigger_interval_steps
        gate_signal = 1 if (is_above_trigger_threshold and enough_interval) else 0
        if gate_signal == 1:
            self._last_trigger_step = self.sample_count

        # 6. 计算置信度（分数离阈值的距离）
        confidence = self._compute_confidence(fused_anomaly_score, adaptive_threshold)

        # 7. 更新模块（关键步骤：在线学习）
        self._update_on_new_sample(
            sample=sample,
            normalized_sample=normalized_sample,
            anomaly_score=fused_anomaly_score,
            gate_decision=gate_signal,
            feedback_label=feedback_label,
        )

        return GateDecision(
            gate_signal=gate_signal,
            anomaly_score=fused_anomaly_score,
            confidence=confidence,
            adaptive_threshold=trigger_threshold,
            timestamp=timestamp,
        )

    def _compute_confidence(self, score: float, threshold: float) -> float:
        """
        计算置信度（分数离阈值的距离）

        分数离阈值越远，置信度越高
        """
        distance_to_threshold = abs(score - threshold)
        # 使用sigmoid函数映射到(0, 1)
        confidence = 1.0 / (1.0 + np.exp(-2 * (distance_to_threshold - 0.1)))
        return float(np.clip(confidence, 0.0, 1.0))

    def _update_on_new_sample(
        self,
        sample: np.ndarray,
        normalized_sample: np.ndarray,
        anomaly_score: float,
        gate_decision: int,
        feedback_label: Optional[int] = None,
    ) -> None:
        """
        用新样本更新所有在线模块（在线学习的关键）

        参数：
            sample: 原始样本
            normalized_sample: 归一化后的样本
            anomaly_score: 计算出的异常分数
            gate_decision: 网关决策
            feedback_label: 人工反馈标签（可选，用于有标签学习）
        """
        # 1. 更新特征统计（EMA）
        self.feature_stats.update(sample)

        # 2. 更新异常检测器参考集
        self.anomaly_detector.update(normalized_sample)

        # 3. 更新阈值优化器
        #    使用反馈标签（如果提供），否则使用网关决策
        decision_for_optimization = feedback_label if feedback_label is not None else gate_decision
        self.threshold_optimizer.update(
            score=anomaly_score,
            decision=decision_for_optimization,
            perform_optimization=True,
        )

        # 4. 更新多尺度分布追踪
        self.multi_scale_tracker.update(normalized_sample)

    def predict_batch(
        self,
        samples: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        feedback_labels: Optional[np.ndarray] = None,
    ) -> list:
        """
        批量预测

        参数：
            samples: 输入样本，shape (n_samples, feature_dim)
            timestamps: 时间戳数组（可选）
            feedback_labels: 反馈标签数组（可选）

        返回：
            GateDecision列表
        """
        decisions = []
        for i, sample in enumerate(samples):
            timestamp = timestamps[i] if timestamps is not None else None
            feedback = feedback_labels[i] if feedback_labels is not None else None
            decision = self.predict(sample, timestamp, feedback)
            decisions.append(decision)

        return decisions

    def get_statistics(self) -> Dict:
        """获取网关的统计信息和诊断数据"""
        return {
            "sample_count": self.sample_count,
            "initialized": self._initialized,
            "score_weights": {
                "short": self.score_short_weight,
                "medium": self.score_medium_weight,
                "long": self.score_long_weight,
            },
            "threshold_bias": self.threshold_bias,
            "trigger_hysteresis_margin": self.trigger_hysteresis_margin,
            "min_trigger_interval_steps": self.min_trigger_interval_steps,
            "last_trigger_step": self._last_trigger_step,
            "feature_stats": self.feature_stats.get_statistics(),
            "anomaly_detector": self.anomaly_detector.get_statistics(),
            "threshold_optimizer": self.threshold_optimizer.get_statistics(),
            "multi_scale_tracker": self.multi_scale_tracker.get_all_statistics(),
        }

    def report_feedback(self, decision_id: int, true_label: int) -> None:
        """
        报告人工反馈（用于优化）

        在生产环境中，可能有延迟的标注反馈
        这个方法可用于集成这些反馈
        """
        # 当前实现简单，可扩展为维护决策历史并进行批量反馈
        pass

    def get_gate_signal(self, sample: np.ndarray) -> int:
        """简化接口：直接获取二值门控信号"""
        decision = self.predict(sample)
        return decision.gate_signal

    def get_anomaly_score(self, sample: np.ndarray) -> float:
        """简化接口：直接获取异常分数"""
        decision = self.predict(sample)
        return decision.anomaly_score

    def save_state(self, path: str | Path) -> None:
        """
        保存网关状态到文件

        参数：
            path: 状态文件路径（.pkl）
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": 2,
            "feature_dim": self.feature_dim,
            "sample_count": self.sample_count,
            "initialized": self._initialized,
            "trigger_hysteresis_margin": self.trigger_hysteresis_margin,
            "min_trigger_interval_steps": self.min_trigger_interval_steps,
            "last_trigger_step": self._last_trigger_step,
            "feature_stats": self.feature_stats,
            "anomaly_detector": self.anomaly_detector,
            "threshold_optimizer": self.threshold_optimizer,
            "multi_scale_tracker": self.multi_scale_tracker,
        }

        with open(target_path, "wb") as file:
            pickle.dump(payload, file)

    def load_state(self, path: str | Path) -> None:
        """
        从文件加载网关状态

        参数：
            path: 状态文件路径（.pkl）
        """
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"网关状态文件不存在: {source_path}")

        with open(source_path, "rb") as file:
            try:
                payload = pickle.load(file)
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "网关状态文件引用了不存在的旧模块路径，请使用当前代码重新生成 gate state 文件后再加载。"
                ) from exc

        saved_feature_dim = int(payload.get("feature_dim", -1))
        if saved_feature_dim != self.feature_dim:
            raise ValueError(
                f"feature_dim 不匹配，当前={self.feature_dim}，状态文件={saved_feature_dim}"
            )

        self.sample_count = int(payload.get("sample_count", 0))
        self._initialized = bool(payload.get("initialized", False))
        self.trigger_hysteresis_margin = float(
            payload.get("trigger_hysteresis_margin", self.trigger_hysteresis_margin)
        )
        self.min_trigger_interval_steps = int(
            payload.get("min_trigger_interval_steps", self.min_trigger_interval_steps)
        )
        self._last_trigger_step = int(payload.get("last_trigger_step", -10**9))
        self.feature_stats = payload["feature_stats"]
        self.anomaly_detector = payload["anomaly_detector"]
        self.threshold_optimizer = payload["threshold_optimizer"]
        self.multi_scale_tracker = payload["multi_scale_tracker"]


