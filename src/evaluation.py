"""
Evaluation and comparison utilities.

Runs each agent/scheduler over multiple episodes with fixed seeds, collects
throughput, latency, fairness, and adaptability metrics, and produces a
comparison table and optional matplotlib charts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.environment import ResourceAllocationEnv
from src.reward import RewardFunction


# ---------------------------------------------------------------------------
# Single-agent evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent,
    env: ResourceAllocationEnv,
    reward_fn: RewardFunction,
    *,
    num_episodes: int = 50,
    seed: int = 42,
    agent_name: str = "agent",
) -> pd.DataFrame:
    """
    Evaluate a trained agent (or baseline) over *num_episodes* and return
    per-episode metrics as a DataFrame.
    """
    records: list[dict] = []

    for ep in range(num_episodes):
        state, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        wait_times: list[float] = []
        node_utils_per_step: list[np.ndarray] = []

        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward_components, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward = reward_fn.compute(reward_components)
            total_reward += reward

            # Collect per-step node utilization for fairness computation
            utils = np.array([
                next_state[i * 5]   # cpu_utilization
                for i in range(env.num_nodes)
            ])
            node_utils_per_step.append(utils)

            # Track average wait from latency component
            if reward_components.get("latency", 0.0) < 0:
                wait_times.append(abs(reward_components["latency"]))

            state = next_state

        # Aggregate episode metrics
        all_utils = np.stack(node_utils_per_step)
        mean_utils = all_utils.mean(axis=0)
        fairness = RewardFunction.jains_fairness(mean_utils)

        records.append({
            "agent": agent_name,
            "episode": ep,
            "total_reward": total_reward,
            "tasks_completed": info.get("tasks_completed", 0),
            "tasks_rejected": info.get("tasks_rejected", 0),
            "throughput": info.get("tasks_completed", 0) / max(info.get("step", 1), 1),
            "avg_latency": float(np.mean(wait_times)) if wait_times else 0.0,
            "fairness": fairness,
            "active_nodes_final": info.get("active_nodes", env.num_nodes),
        })

        agent.on_episode_end(ep)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Multi-agent comparison
# ---------------------------------------------------------------------------

def compare_agents(
    agents: dict[str, object],
    env: ResourceAllocationEnv,
    reward_fn: RewardFunction,
    *,
    num_episodes: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate every agent in *agents* dict and return a combined DataFrame.
    """
    frames: list[pd.DataFrame] = []
    for name, agent in agents.items():
        print(f"Evaluating {name} ...")
        df = evaluate_agent(
            agent, env, reward_fn,
            num_episodes=num_episodes,
            seed=seed,
            agent_name=name,
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a summary table with mean and std for each metric grouped by
    agent name.
    """
    metrics = [
        "total_reward",
        "tasks_completed",
        "tasks_rejected",
        "throughput",
        "avg_latency",
        "fairness",
    ]
    agg = results.groupby("agent")[metrics].agg(["mean", "std"])
    # Flatten multi-level columns
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.round(4)


# ---------------------------------------------------------------------------
# Adaptability analysis
# ---------------------------------------------------------------------------

def adaptability_analysis(
    agents: dict[str, object],
    env_config: dict,
    reward_fn: RewardFunction,
    *,
    num_episodes: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate agents under increasingly harsh dynamic conditions (higher
    failure rates) to measure adaptability.
    """
    failure_rates = [0.0, 0.005, 0.01, 0.02, 0.05]
    records: list[dict] = []

    for fail_rate in failure_rates:
        cfg = dict(env_config)
        dynamics = dict(cfg.get("dynamics", {}))
        dynamics["node_failure_prob"] = fail_rate
        cfg["dynamics"] = dynamics
        env = ResourceAllocationEnv(cfg)

        for name, agent in agents.items():
            df = evaluate_agent(
                agent, env, reward_fn,
                num_episodes=num_episodes,
                seed=seed,
                agent_name=name,
            )
            records.append({
                "agent": name,
                "failure_rate": fail_rate,
                "mean_reward": df["total_reward"].mean(),
                "mean_throughput": df["throughput"].mean(),
                "mean_fairness": df["fairness"].mean(),
                "mean_completed": df["tasks_completed"].mean(),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results: pd.DataFrame, save_dir: str = "results"):
    """Generate bar charts comparing agents across key metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        "total_reward": "Total Reward",
        "tasks_completed": "Tasks Completed",
        "throughput": "Throughput (tasks/step)",
        "avg_latency": "Average Latency",
        "fairness": "Fairness (Jain's Index)",
    }

    agent_names = results["agent"].unique()

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics.items()):
        means = [results[results["agent"] == a][col].mean() for a in agent_names]
        stds = [results[results["agent"] == a][col].std() for a in agent_names]
        bars = ax.bar(agent_names, means, yerr=stds, capsize=4, alpha=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(label)
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{m:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    fig.savefig(save_path / "comparison.png", dpi=150)
    plt.close(fig)
    print(f"Comparison chart saved to {save_path / 'comparison.png'}")


def plot_training_curves(
    metrics_dict: dict[str, list[dict]],
    save_dir: str = "results",
):
    """Plot reward curves from training logs for each agent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, metrics in metrics_dict.items():
        episodes = [m["episode"] for m in metrics]
        rewards = [m["reward"] for m in metrics]
        # Smoothed curve (rolling mean window=20)
        window = min(20, len(rewards))
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), smoothed, label=name)
        else:
            ax.plot(episodes, rewards, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Reward Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_path / 'training_curves.png'}")


def plot_adaptability(adapt_df: pd.DataFrame, save_dir: str = "results"):
    """Plot throughput vs failure rate for each agent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for agent_name in adapt_df["agent"].unique():
        subset = adapt_df[adapt_df["agent"] == agent_name]
        ax.plot(
            subset["failure_rate"],
            subset["mean_throughput"],
            marker="o",
            label=agent_name,
        )

    ax.set_xlabel("Node Failure Probability")
    ax.set_ylabel("Mean Throughput (tasks/step)")
    ax.set_title("Adaptability: Throughput Under Increasing Failures")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path / "adaptability.png", dpi=150)
    plt.close(fig)
    print(f"Adaptability chart saved to {save_path / 'adaptability.png'}")
