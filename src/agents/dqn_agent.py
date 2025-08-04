"""
Deep Q-Network (DQN) agent for resource allocation.

Features:
    - Experience replay buffer
    - Separate target network with periodic hard updates
    - Epsilon-greedy exploration with decay
    - PyTorch-based Q-network
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Fully connected Q-value network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: dict | None = None,
    ):
        cfg = config or {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma: float = cfg.get("gamma", 0.99)
        self.lr: float = cfg.get("learning_rate", 1e-3)
        self.epsilon: float = cfg.get("epsilon_start", 1.0)
        self.epsilon_end: float = cfg.get("epsilon_end", 0.05)
        self.epsilon_decay: float = cfg.get("epsilon_decay", 0.995)
        self.batch_size: int = cfg.get("batch_size", 64)
        self.target_update_freq: int = cfg.get("target_update_freq", 10)
        hidden_dims: list[int] = cfg.get("hidden_dims", [128, 128])
        buffer_size: int = cfg.get("buffer_size", 50_000)

        # Networks
        self.q_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.update_count: int = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(s)
            return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        """Sample a mini-batch and perform one gradient step. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([float(t.done) for t in batch]).unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions)

        # Target Q-values
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Hard copy of Q-net weights to target net."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # End-of-episode hook (called by training loop)
    # ------------------------------------------------------------------

    def on_episode_end(self, episode: int):
        """Decay epsilon and optionally sync the target network."""
        self.decay_epsilon()
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
