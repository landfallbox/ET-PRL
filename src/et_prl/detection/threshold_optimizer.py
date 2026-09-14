"""
流式阈值优化模块 - 在线动态调整异常判断阈值
"""

from __future__ import annotations

from collections import deque

import numpy as np


class StreamingThresholdOptimizer:
    """在线流式阈值优化"""

    def __init__(
        self,
        local_window_size: int = 100,
        global_ema_decay: float = 0.01,
        alpha: float = 0.7,
        min_samples_for_optimization: int = 50,
        quantile: float = 0.8,
        mad_scale: float = 1.8,
        local_update_rate: float = 0.3,
        quantile_weight: float = 0.6,
    ):
        self.local_window_size = local_window_size
        self.global_ema_decay = global_ema_decay
        self.alpha = alpha
        self.min_samples_for_optimization = min_samples_for_optimization
        self.quantile = quantile
        self.mad_scale = mad_scale
        self.local_update_rate = local_update_rate
        self.quantile_weight = quantile_weight

        self.local_score_buffer = deque(maxlen=local_window_size)

        self.global_threshold = 0.5
        self.local_threshold = 0.5
        self.adaptive_threshold = 0.5

        self.sample_count = 0
        self.local_candidate_history = deque(maxlen=1000)

    def _robust_mad_threshold(self, scores: np.ndarray) -> float:
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        robust_std = 1.4826 * mad
        return median + self.mad_scale * robust_std

    def _optimize_threshold_unsupervised(self) -> float:
        if len(self.local_score_buffer) < self.min_samples_for_optimization:
            return self.local_threshold

        scores = np.array(list(self.local_score_buffer))

        quantile_threshold = float(np.quantile(scores, self.quantile))
        mad_threshold = self._robust_mad_threshold(scores)

        candidate = self.quantile_weight * quantile_threshold + (1.0 - self.quantile_weight) * mad_threshold
        candidate = float(np.clip(candidate, 0.0, 1.0))
        self.local_candidate_history.append(candidate)

        return (1.0 - self.local_update_rate) * self.local_threshold + self.local_update_rate * candidate

    def update(self, score: float, decision: int | None = None, perform_optimization: bool = True) -> None:
        self.sample_count += 1

        self.local_score_buffer.append(score)

        if perform_optimization and self.sample_count % 10 == 0:
            self.local_threshold = self._optimize_threshold_unsupervised()

        if self.sample_count == 1:
            self.global_threshold = self.local_threshold
        else:
            self.global_threshold = (
                (1 - self.global_ema_decay) * self.global_threshold
                + self.global_ema_decay * self.local_threshold
            )

        self.adaptive_threshold = (
            self.alpha * self.local_threshold + (1 - self.alpha) * self.global_threshold
        )

    def get_adaptive_threshold(self) -> float:
        return self.adaptive_threshold

    def get_local_threshold(self) -> float:
        return self.local_threshold

    def get_global_threshold(self) -> float:
        return self.global_threshold

    def get_statistics(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "adaptive_threshold": self.adaptive_threshold,
            "local_threshold": self.local_threshold,
            "global_threshold": self.global_threshold,
            "local_buffer_size": len(self.local_score_buffer),
            "quantile": self.quantile,
            "mad_scale": self.mad_scale,
            "local_update_rate": self.local_update_rate,
            "quantile_weight": self.quantile_weight,
            "recent_local_candidate": float(self.local_candidate_history[-1])
            if len(self.local_candidate_history) > 0
            else 0.0,
            "avg_local_candidate": float(np.mean(list(self.local_candidate_history)))
            if len(self.local_candidate_history) > 0
            else 0.0,
        }
