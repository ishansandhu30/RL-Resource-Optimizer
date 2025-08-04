"""
Proximal Policy Optimization (PPO) agent for resource allocation.

Features:
    - Actor-critic architecture with shared feature extractor
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Entropy bonus for exploration
    - PyTorch-based
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network with separate policy and value heads."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        # Shared feature extractor
        trunk_layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)

        # Policy head (actor)
        self.policy_head = nn.Linear(prev, action_dim)

        # Value head (critic)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor):
        features = self.trunk(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores transitions for a single rollout, then computes GAE returns."""

    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns."""
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return advantages, returns

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.states)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """PPO agent with actor-critic, GAE, and clipped surrogate loss."""

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
        self.gae_lambda: float = cfg.get("gae_lambda", 0.95)
        self.clip_eps: float = cfg.get("clip_epsilon", 0.2)
        self.entropy_coef: float = cfg.get("entropy_coef", 0.01)
        self.value_coef: float = cfg.get("value_loss_coef", 0.5)
        self.max_grad_norm: float = cfg.get("max_grad_norm", 0.5)
        self.update_epochs: int = cfg.get("update_epochs", 4)
        self.mini_batch_size: int = cfg.get("mini_batch_size", 64)
        self.lr: float = cfg.get("learning_rate", 3e-4)
        hidden_dims: list[int] = cfg.get("hidden_dims", [128, 128])

        # Network
        self.ac = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)

        # Rollout storage
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.ac(s)
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        self._last_log_prob = log_prob.item()
        self._last_value = value.item()
        return action.item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store a transition in the rollout buffer."""
        self.buffer.store(
            state=state,
            action=action,
            reward=reward,
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> float | None:
        """Run PPO update over the collected rollout. Returns mean policy loss."""
        if len(self.buffer) == 0:
            return None

        # Bootstrap value for last state
        last_state = self.buffer.states[-1]
        with torch.no_grad():
            s = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            _, last_val = self.ac(s)
            last_value = last_val.item()

        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions_t = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss = 0.0
        n_updates = 0
        n = len(self.buffer)

        for _ in range(self.update_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                mb_idx = indices[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_lp = old_log_probs_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_ret = returns_t[mb_idx]

                _, new_lp, entropy, values = self.ac.get_action_and_value(
                    mb_states, mb_actions
                )

                # Clipped surrogate objective
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_ret)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += policy_loss.item()
                n_updates += 1

        self.buffer.clear()
        return total_loss / max(n_updates, 1)

    def on_episode_end(self, episode: int):
        """PPO updates happen via explicit update() calls, so this is a no-op."""
        pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ac": self.ac.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.ac.load_state_dict(ckpt["ac"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
