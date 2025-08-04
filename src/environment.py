"""
Custom Gymnasium environment for dynamic resource allocation.

Models a cluster of compute nodes receiving a stream of tasks with varying
CPU and memory demands.  An agent must decide which node handles each incoming
task while the system experiences fluctuating workloads, node failures, and
capacity changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A unit of work that must be placed on a node."""
    task_id: int
    cpu_demand: float
    memory_demand: float
    duration: int            # remaining steps until completion
    arrival_step: int        # step when the task entered the queue
    wait_time: int = 0       # steps spent waiting in the queue


@dataclass
class Node:
    """A single compute node in the cluster."""
    node_id: int
    cpu_capacity: float
    memory_capacity: float
    cpu_used: float = 0.0
    memory_used: float = 0.0
    active: bool = True
    running_tasks: list[Task] = field(default_factory=list)

    @property
    def cpu_available(self) -> float:
        return max(0.0, self.cpu_capacity - self.cpu_used) if self.active else 0.0

    @property
    def memory_available(self) -> float:
        return max(0.0, self.memory_capacity - self.memory_used) if self.active else 0.0

    @property
    def cpu_utilization(self) -> float:
        return self.cpu_used / self.cpu_capacity if self.cpu_capacity > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        return self.memory_used / self.memory_capacity if self.memory_capacity > 0 else 0.0


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class ResourceAllocationEnv(gym.Env):
    """
    Observation (flat vector per node + leading task info):
        For each node  : [cpu_util, mem_util, cpu_avail, mem_avail, active]
        Task queue head : [cpu_demand, mem_demand, duration, wait_time]
        Queue metadata  : [queue_length_normalized]

    Action:
        Discrete(num_nodes) -- index of the node to assign the head-of-queue
        task to.  If the chosen node cannot fit the task the action is treated
        as a rejection (with penalty).

    Dynamics:
        * Tasks arrive stochastically each step.
        * Running tasks tick down; completed tasks free resources.
        * Nodes may fail or recover probabilistically.
        * Node capacities fluctuate slightly each step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}

        # Environment parameters
        self.num_nodes: int = cfg.get("num_nodes", 5)
        self.node_cpu_cap: float = cfg.get("node_capacity_cpu", 100.0)
        self.node_mem_cap: float = cfg.get("node_capacity_memory", 64.0)
        self.max_queue: int = cfg.get("max_queue_size", 20)
        self.max_steps: int = cfg.get("max_steps", 500)

        # Task arrival parameters
        ta = cfg.get("task_arrival", {})
        self.arrival_rate: float = ta.get("rate", 0.7)
        self.cpu_range: tuple = tuple(ta.get("cpu_range", [5, 40]))
        self.mem_range: tuple = tuple(ta.get("memory_range", [1, 16]))
        self.dur_range: tuple = tuple(ta.get("duration_range", [3, 20]))

        # Dynamic constraint parameters
        dyn = cfg.get("dynamics", {})
        self.fail_prob: float = dyn.get("node_failure_prob", 0.005)
        self.recover_prob: float = dyn.get("node_recovery_prob", 0.02)
        self.cap_fluct: float = dyn.get("capacity_fluctuation", 0.05)

        # Spaces
        # Observation: 5 features per node + 4 task features + 1 queue length
        obs_size = self.num_nodes * 5 + 4 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_nodes)

        # Internal state (initialized in reset)
        self.nodes: list[Node] = []
        self.task_queue: list[Task] = []
        self.current_step: int = 0
        self._task_counter: int = 0

        # Metrics tracked per episode
        self.tasks_completed: int = 0
        self.tasks_rejected: int = 0
        self.total_wait_time: float = 0.0
        self.total_tasks_seen: int = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.nodes = [
            Node(
                node_id=i,
                cpu_capacity=self.node_cpu_cap,
                memory_capacity=self.node_mem_cap,
            )
            for i in range(self.num_nodes)
        ]
        self.task_queue = []
        self.current_step = 0
        self._task_counter = 0
        self.tasks_completed = 0
        self.tasks_rejected = 0
        self.total_wait_time = 0.0
        self.total_tasks_seen = 0

        # Seed the queue with an initial task so there's always a decision
        self._maybe_generate_tasks(force=True)

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        reward_components: dict[str, float] = {
            "throughput": 0.0,
            "latency": 0.0,
            "fairness": 0.0,
            "rejection": 0.0,
        }

        # --- 1. Try to place the head-of-queue task on the chosen node ---
        placed = False
        if self.task_queue:
            task = self.task_queue[0]
            node = self.nodes[action]
            if (
                node.active
                and node.cpu_available >= task.cpu_demand
                and node.memory_available >= task.memory_demand
            ):
                # Place the task
                node.cpu_used += task.cpu_demand
                node.memory_used += task.memory_demand
                node.running_tasks.append(task)
                self.task_queue.pop(0)
                placed = True
                # Latency component: lower wait = better
                self.total_wait_time += task.wait_time
            else:
                # Rejection: node cannot accept the task
                self.tasks_rejected += 1
                self.task_queue.pop(0)
                reward_components["rejection"] = -1.0

        # --- 2. Tick running tasks, free completed ones ---
        completed_this_step = 0
        for node in self.nodes:
            still_running = []
            for t in node.running_tasks:
                t.duration -= 1
                if t.duration <= 0:
                    node.cpu_used = max(0.0, node.cpu_used - t.cpu_demand)
                    node.memory_used = max(0.0, node.memory_used - t.memory_demand)
                    completed_this_step += 1
                else:
                    still_running.append(t)
            node.running_tasks = still_running
        self.tasks_completed += completed_this_step
        reward_components["throughput"] = float(completed_this_step)

        # --- 3. Increment wait times in queue ---
        for t in self.task_queue:
            t.wait_time += 1

        # Average normalized wait penalty for tasks still in queue
        if self.task_queue:
            avg_wait = np.mean([t.wait_time for t in self.task_queue])
            reward_components["latency"] = -avg_wait / self.max_steps
        else:
            reward_components["latency"] = 0.0

        # --- 4. Fairness across node utilizations (Jain's index) ---
        utils = np.array(
            [n.cpu_utilization for n in self.nodes if n.active], dtype=np.float32
        )
        if len(utils) > 0 and np.sum(utils) > 0:
            jains = (np.sum(utils) ** 2) / (len(utils) * np.sum(utils ** 2))
        else:
            jains = 1.0  # trivially fair when idle
        reward_components["fairness"] = jains  # in [0, 1]

        # --- 5. Dynamic environment changes ---
        self._apply_dynamics()

        # --- 6. Generate new tasks ---
        self._maybe_generate_tasks()

        # --- 7. Drop overflow tasks ---
        while len(self.task_queue) > self.max_queue:
            self.task_queue.pop(-1)
            self.tasks_rejected += 1
            reward_components["rejection"] -= 0.5

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        info["reward_components"] = reward_components

        return obs, reward_components, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = []
        for node in self.nodes:
            obs.extend([
                node.cpu_utilization,
                node.memory_utilization,
                node.cpu_available / self.node_cpu_cap,
                node.memory_available / self.node_mem_cap,
                float(node.active),
            ])
        # Head-of-queue task features (normalized)
        if self.task_queue:
            t = self.task_queue[0]
            obs.extend([
                t.cpu_demand / self.cpu_range[1],
                t.memory_demand / self.mem_range[1],
                t.duration / self.dur_range[1],
                min(t.wait_time / 50.0, 1.0),
            ])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])
        # Queue length
        obs.append(len(self.task_queue) / self.max_queue)
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        active = sum(1 for n in self.nodes if n.active)
        return {
            "step": self.current_step,
            "tasks_completed": self.tasks_completed,
            "tasks_rejected": self.tasks_rejected,
            "queue_length": len(self.task_queue),
            "active_nodes": active,
            "total_tasks_seen": self.total_tasks_seen,
        }

    def _maybe_generate_tasks(self, force: bool = False):
        if force or self.np_random.random() < self.arrival_rate:
            task = Task(
                task_id=self._task_counter,
                cpu_demand=self.np_random.uniform(*self.cpu_range),
                memory_demand=self.np_random.uniform(*self.mem_range),
                duration=self.np_random.integers(self.dur_range[0], self.dur_range[1] + 1),
                arrival_step=self.current_step,
            )
            self._task_counter += 1
            self.total_tasks_seen += 1
            if len(self.task_queue) < self.max_queue:
                self.task_queue.append(task)

    def _apply_dynamics(self):
        """Simulate node failures, recoveries, and capacity fluctuations."""
        for node in self.nodes:
            # Failures and recoveries
            if node.active and self.np_random.random() < self.fail_prob:
                node.active = False
                # Evict running tasks back to queue head
                for t in node.running_tasks:
                    t.duration = max(1, t.duration)
                    if len(self.task_queue) < self.max_queue:
                        self.task_queue.insert(0, t)
                node.running_tasks = []
                node.cpu_used = 0.0
                node.memory_used = 0.0
            elif not node.active and self.np_random.random() < self.recover_prob:
                node.active = True

            # Capacity fluctuations (only active nodes)
            if node.active:
                delta_cpu = self.np_random.uniform(
                    -self.cap_fluct, self.cap_fluct
                ) * self.node_cpu_cap
                delta_mem = self.np_random.uniform(
                    -self.cap_fluct, self.cap_fluct
                ) * self.node_mem_cap
                node.cpu_capacity = np.clip(
                    node.cpu_capacity + delta_cpu,
                    self.node_cpu_cap * 0.7,
                    self.node_cpu_cap * 1.3,
                )
                node.memory_capacity = np.clip(
                    node.memory_capacity + delta_mem,
                    self.node_mem_cap * 0.7,
                    self.node_mem_cap * 1.3,
                )
                # Clamp usage to not exceed new capacity
                node.cpu_used = min(node.cpu_used, node.cpu_capacity)
                node.memory_used = min(node.memory_used, node.memory_capacity)
