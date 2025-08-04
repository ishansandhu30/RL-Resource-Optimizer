"""
Rule-based baseline schedulers for comparison against RL agents.

Each scheduler exposes the same interface as the RL agents:
    - select_action(state, deterministic=True) -> int
    - store_transition(...)   (no-op)
    - update()                (no-op)
    - on_episode_end(...)     (no-op)

This makes them drop-in replacements inside the training and evaluation loops.
"""

from __future__ import annotations

import numpy as np


class _BaseScheduler:
    """Shared no-op methods so baselines plug into the training loop."""

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def store_transition(self, state, action, reward, next_state, done):
        pass

    def update(self) -> float | None:
        return None

    def on_episode_end(self, episode: int):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class RoundRobinScheduler(_BaseScheduler):
    """Assigns tasks to nodes in a fixed cyclic order, skipping inactive nodes."""

    def __init__(self, num_nodes: int):
        super().__init__(num_nodes)
        self._idx = 0

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        # Decode which nodes are active from the observation vector.
        # State layout per node: [cpu_util, mem_util, cpu_avail, mem_avail, active]
        active_mask = self._get_active_mask(state)

        # Find the next active node in round-robin order
        for _ in range(self.num_nodes):
            candidate = self._idx % self.num_nodes
            self._idx += 1
            if active_mask[candidate]:
                return candidate
        # Fallback: pick any node
        return self._idx % self.num_nodes

    def on_episode_end(self, episode: int):
        self._idx = 0

    def _get_active_mask(self, state: np.ndarray) -> list[bool]:
        return [bool(state[i * 5 + 4] > 0.5) for i in range(self.num_nodes)]


class GreedyScheduler(_BaseScheduler):
    """
    Assigns each task to the *least-loaded* active node (by CPU utilization).
    This is a strong heuristic baseline that favours balanced load.
    """

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        best_node = 0
        best_avail = -1.0
        for i in range(self.num_nodes):
            offset = i * 5
            active = state[offset + 4] > 0.5
            cpu_avail = state[offset + 2]  # normalized available CPU
            if active and cpu_avail > best_avail:
                best_avail = cpu_avail
                best_node = i
        return best_node


class RandomScheduler(_BaseScheduler):
    """Assigns tasks uniformly at random to active nodes."""

    def __init__(self, num_nodes: int, seed: int | None = None):
        super().__init__(num_nodes)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        active_nodes = [
            i for i in range(self.num_nodes)
            if state[i * 5 + 4] > 0.5
        ]
        if not active_nodes:
            return self.rng.integers(0, self.num_nodes)
        return int(self.rng.choice(active_nodes))
