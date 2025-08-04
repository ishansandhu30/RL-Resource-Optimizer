"""
Composite reward function for the resource-allocation environment.

Combines throughput, latency, and fairness signals with configurable weights,
plus discrete penalties/bonuses for task rejections and completions.
"""

from __future__ import annotations

import numpy as np


class RewardFunction:
    """Compute a scalar reward from per-step component signals."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.w_throughput: float = cfg.get("throughput_weight", 0.4)
        self.w_latency: float = cfg.get("latency_weight", 0.35)
        self.w_fairness: float = cfg.get("fairness_weight", 0.25)
        self.rejection_penalty: float = cfg.get("rejection_penalty", -2.0)
        self.completion_bonus: float = cfg.get("completion_bonus", 1.0)

    def compute(self, components: dict[str, float]) -> float:
        """
        Combine raw reward components into a single scalar.

        Parameters
        ----------
        components : dict
            Keys produced by ``ResourceAllocationEnv.step``:
            - throughput : number of tasks completed this step
            - latency   : negative normalized average wait time
            - fairness  : Jain's fairness index in [0, 1]
            - rejection : negative count-based penalty for rejected tasks

        Returns
        -------
        float
            Weighted scalar reward.
        """
        throughput = components.get("throughput", 0.0)
        latency = components.get("latency", 0.0)
        fairness = components.get("fairness", 1.0)
        rejection = components.get("rejection", 0.0)

        reward = (
            self.w_throughput * (throughput * self.completion_bonus)
            + self.w_latency * latency * 10.0   # scale up to match other terms
            + self.w_fairness * fairness
            + rejection * abs(self.rejection_penalty)
        )
        return float(reward)

    @staticmethod
    def jains_fairness(utilizations: np.ndarray) -> float:
        """Compute Jain's fairness index over an array of utilization values."""
        if len(utilizations) == 0 or np.sum(utilizations) == 0:
            return 1.0
        n = len(utilizations)
        return float((np.sum(utilizations) ** 2) / (n * np.sum(utilizations ** 2)))

    def __repr__(self) -> str:
        return (
            f"RewardFunction(throughput={self.w_throughput}, "
            f"latency={self.w_latency}, fairness={self.w_fairness})"
        )
