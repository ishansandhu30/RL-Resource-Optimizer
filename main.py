"""
RL-Resource-Optimizer -- CLI entry point.

Modes
-----
    train    : Train DQN and/or PPO agents.
    evaluate : Evaluate a saved agent checkpoint.
    compare  : Run all agents + baselines and produce comparison artefacts.

Usage
-----
    python main.py train   --agent dqn --episodes 300
    python main.py train   --agent ppo --episodes 300
    python main.py train   --agent both
    python main.py evaluate --agent dqn --checkpoint results/checkpoints/dqn_best.pt
    python main.py compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from src.agents.baseline_schedulers import (
    GreedyScheduler,
    RandomScheduler,
    RoundRobinScheduler,
)
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.environment import ResourceAllocationEnv
from src.evaluation import (
    adaptability_analysis,
    compare_agents,
    plot_adaptability,
    plot_comparison,
    plot_training_curves,
    summary_table,
)
from src.reward import RewardFunction
from src.training import train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(cfg: dict) -> ResourceAllocationEnv:
    return ResourceAllocationEnv(cfg.get("environment", {}))


def make_reward(cfg: dict) -> RewardFunction:
    return RewardFunction(cfg.get("reward", {}))


def _obs_dim(env: ResourceAllocationEnv) -> int:
    return env.observation_space.shape[0]


def _act_dim(env: ResourceAllocationEnv) -> int:
    return env.action_space.n


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train(args, cfg: dict):
    env = make_env(cfg)
    reward_fn = make_reward(cfg)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    training_curves: dict[str, list[dict]] = {}
    agents_to_train = []

    if args.agent in ("dqn", "both"):
        agents_to_train.append(("dqn", DQNAgent(
            _obs_dim(env), _act_dim(env), cfg.get("dqn", {})
        )))
    if args.agent in ("ppo", "both"):
        agents_to_train.append(("ppo", PPOAgent(
            _obs_dim(env), _act_dim(env), cfg.get("ppo", {})
        )))

    for name, agent in agents_to_train:
        episodes = args.episodes or cfg.get(name, {}).get("num_episodes", 500)
        print(f"\n{'='*60}")
        print(f" Training {name.upper()} for {episodes} episodes")
        print(f"{'='*60}")
        metrics = train(
            agent,
            env,
            reward_fn,
            num_episodes=episodes,
            checkpoint_dir=str(results_dir / "checkpoints"),
            checkpoint_every=max(1, episodes // 10),
            log_path=str(results_dir / f"{name}_training_log.jsonl"),
            agent_name=name,
        )
        training_curves[name.upper()] = metrics

    if training_curves:
        plot_training_curves(training_curves, save_dir=str(results_dir))
        print("\nTraining complete. Checkpoints and logs saved to results/")


def cmd_evaluate(args, cfg: dict):
    env = make_env(cfg)
    reward_fn = make_reward(cfg)
    num_episodes = cfg.get("evaluation", {}).get("num_eval_episodes", 50)
    seed = cfg.get("evaluation", {}).get("seed", 42)

    agent_name = args.agent
    if agent_name == "dqn":
        agent = DQNAgent(_obs_dim(env), _act_dim(env), cfg.get("dqn", {}))
    elif agent_name == "ppo":
        agent = PPOAgent(_obs_dim(env), _act_dim(env), cfg.get("ppo", {}))
    else:
        print(f"Unknown agent: {agent_name}")
        sys.exit(1)

    ckpt = args.checkpoint
    if not ckpt:
        ckpt = f"results/checkpoints/{agent_name}_best.pt"
    if not Path(ckpt).exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    agent.load(ckpt)
    print(f"Loaded {agent_name.upper()} from {ckpt}")

    from src.evaluation import evaluate_agent
    df = evaluate_agent(
        agent, env, reward_fn,
        num_episodes=num_episodes,
        seed=seed,
        agent_name=agent_name.upper(),
    )

    print(f"\n{agent_name.upper()} evaluation over {num_episodes} episodes:")
    print(f"  Mean reward:     {df['total_reward'].mean():+.2f} +/- {df['total_reward'].std():.2f}")
    print(f"  Tasks completed: {df['tasks_completed'].mean():.1f} +/- {df['tasks_completed'].std():.1f}")
    print(f"  Throughput:      {df['throughput'].mean():.4f}")
    print(f"  Avg latency:     {df['avg_latency'].mean():.4f}")
    print(f"  Fairness:        {df['fairness'].mean():.4f}")


def cmd_compare(args, cfg: dict):
    env = make_env(cfg)
    reward_fn = make_reward(cfg)
    num_nodes = cfg.get("environment", {}).get("num_nodes", 5)
    num_episodes = cfg.get("evaluation", {}).get("num_eval_episodes", 50)
    seed = cfg.get("evaluation", {}).get("seed", 42)
    results_dir = Path("results")

    # Build agents dict
    agents: dict[str, object] = {}

    # RL agents -- load best checkpoints if they exist
    dqn_path = results_dir / "checkpoints" / "dqn_best.pt"
    if dqn_path.exists():
        dqn = DQNAgent(_obs_dim(env), _act_dim(env), cfg.get("dqn", {}))
        dqn.load(dqn_path)
        agents["DQN"] = dqn
        print(f"Loaded DQN from {dqn_path}")

    ppo_path = results_dir / "checkpoints" / "ppo_best.pt"
    if ppo_path.exists():
        ppo = PPOAgent(_obs_dim(env), _act_dim(env), cfg.get("ppo", {}))
        ppo.load(ppo_path)
        agents["PPO"] = ppo
        print(f"Loaded PPO from {ppo_path}")

    # Baselines
    agents["RoundRobin"] = RoundRobinScheduler(num_nodes)
    agents["Greedy"] = GreedyScheduler(num_nodes)
    agents["Random"] = RandomScheduler(num_nodes, seed=seed)

    if not agents:
        print("No agents to evaluate.")
        sys.exit(1)

    # Standard comparison
    print(f"\nComparing {len(agents)} agents over {num_episodes} episodes ...\n")
    results = compare_agents(
        agents, env, reward_fn,
        num_episodes=num_episodes,
        seed=seed,
    )
    results.to_csv(results_dir / "comparison_results.csv", index=False)

    table = summary_table(results)
    print("\n" + "=" * 80)
    print("  COMPARISON SUMMARY")
    print("=" * 80)
    print(table.to_string())
    print()

    # Adaptability analysis
    print("Running adaptability analysis ...")
    env_cfg = cfg.get("environment", {})
    adapt_df = adaptability_analysis(
        agents, env_cfg, reward_fn,
        num_episodes=min(20, num_episodes),
        seed=seed,
    )
    adapt_df.to_csv(results_dir / "adaptability_results.csv", index=False)

    print("\n" + "=" * 80)
    print("  ADAPTABILITY (throughput under increasing failure rates)")
    print("=" * 80)
    pivot = adapt_df.pivot(
        index="failure_rate", columns="agent", values="mean_throughput"
    )
    print(pivot.round(4).to_string())
    print()

    # Generate plots
    plot_comparison(results, save_dir=str(results_dir))
    plot_adaptability(adapt_df, save_dir=str(results_dir))

    print("All results saved to results/")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RL-Resource-Optimizer: Train and evaluate RL agents for "
                    "dynamic resource allocation.",
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train an RL agent")
    p_train.add_argument(
        "--agent", choices=["dqn", "ppo", "both"], default="both",
        help="Which agent(s) to train",
    )
    p_train.add_argument(
        "--episodes", type=int, default=None,
        help="Override number of training episodes",
    )

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate a saved checkpoint")
    p_eval.add_argument("--agent", choices=["dqn", "ppo"], required=True)
    p_eval.add_argument("--checkpoint", type=str, default=None)

    # compare
    sub.add_parser("compare", help="Compare RL agents against baselines")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    cfg = load_config(args.config)

    # Set global seeds for reproducibility
    seed = cfg.get("evaluation", {}).get("seed", 42)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    {"train": cmd_train, "evaluate": cmd_evaluate, "compare": cmd_compare}[
        args.command
    ](args, cfg)


if __name__ == "__main__":
    main()
