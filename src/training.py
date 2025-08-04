"""
Training loop for RL agents in the resource-allocation environment.

Supports both DQN (step-level updates) and PPO (rollout-level updates) via
a unified interface.  Logs episode-level metrics and saves periodic
checkpoints.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.environment import ResourceAllocationEnv
from src.reward import RewardFunction


def train(
    agent,
    env: ResourceAllocationEnv,
    reward_fn: RewardFunction,
    *,
    num_episodes: int = 500,
    checkpoint_dir: str = "results/checkpoints",
    checkpoint_every: int = 50,
    log_path: str | None = None,
    agent_name: str = "agent",
    verbose: bool = True,
) -> list[dict]:
    """
    Run the full training loop.

    Parameters
    ----------
    agent : DQNAgent | PPOAgent
        Any agent that implements select_action, store_transition, update,
        and on_episode_end.
    env : ResourceAllocationEnv
        The Gymnasium environment.
    reward_fn : RewardFunction
        Converts raw reward components into a scalar signal.
    num_episodes : int
        Total training episodes.
    checkpoint_dir : str
        Where to save model checkpoints.
    checkpoint_every : int
        Save a checkpoint every N episodes.
    log_path : str | None
        If set, write JSON-lines episode logs to this file.
    agent_name : str
        Label used in log messages and checkpoint filenames.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    list[dict]
        Per-episode metrics.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    if log_path:
        log_file = open(log_path, "w")

    all_metrics: list[dict] = []
    best_reward = -float("inf")
    t_start = time.time()

    # Detect agent type for update strategy
    agent_type = type(agent).__name__  # "DQNAgent" or "PPOAgent"
    is_ppo = "PPO" in agent_type

    for ep in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        episode_losses: list[float] = []
        step = 0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward_components, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = reward_fn.compute(reward_components)
            agent.store_transition(state, action, reward, next_state, done)

            # DQN learns every step; PPO collects then learns in batch
            if not is_ppo:
                loss = agent.update()
                if loss is not None:
                    episode_losses.append(loss)

            episode_reward += reward
            state = next_state
            step += 1

        # PPO: run update at end of episode
        if is_ppo:
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

        agent.on_episode_end(ep)

        # Collect metrics
        metrics = {
            "episode": ep,
            "reward": round(episode_reward, 4),
            "steps": step,
            "tasks_completed": info.get("tasks_completed", 0),
            "tasks_rejected": info.get("tasks_rejected", 0),
            "queue_length": info.get("queue_length", 0),
            "active_nodes": info.get("active_nodes", 0),
            "mean_loss": round(float(np.mean(episode_losses)), 6) if episode_losses else None,
        }
        if hasattr(agent, "epsilon"):
            metrics["epsilon"] = round(agent.epsilon, 4)
        all_metrics.append(metrics)

        if log_file:
            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()

        # Checkpointing
        if ep % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"{agent_name}_ep{ep}.pt"
            agent.save(ckpt_path)
            if verbose:
                elapsed = time.time() - t_start
                print(
                    f"[{agent_name}] Episode {ep}/{num_episodes} | "
                    f"Reward: {episode_reward:+.2f} | "
                    f"Completed: {metrics['tasks_completed']} | "
                    f"Rejected: {metrics['tasks_rejected']} | "
                    f"Time: {elapsed:.1f}s"
                )

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(ckpt_dir / f"{agent_name}_best.pt")

    # Save final model
    agent.save(ckpt_dir / f"{agent_name}_final.pt")

    if log_file:
        log_file.close()

    if verbose:
        elapsed = time.time() - t_start
        print(
            f"\n[{agent_name}] Training complete: {num_episodes} episodes in {elapsed:.1f}s"
        )
        rewards = [m["reward"] for m in all_metrics]
        print(
            f"  Mean reward (last 50): {np.mean(rewards[-50:]):+.2f} | "
            f"Best: {best_reward:+.2f}"
        )

    return all_metrics
