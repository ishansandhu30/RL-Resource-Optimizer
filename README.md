# RL-Resource-Optimizer

A reinforcement learning framework for optimizing dynamic resource allocation across competing tasks under changing system constraints.

## Overview

This project models cluster resource allocation as a sequential decision process. Incoming tasks with varying CPU and memory demands must be assigned to compute nodes in real time, while the system experiences fluctuating workloads, node failures, and capacity changes.

Two RL agents (DQN and PPO) are trained with custom reward functions that balance throughput, latency, and fairness (Jain's index). Their learned policies are evaluated against rule-based schedulers (round-robin, greedy least-loaded, random) across standard and stress-test scenarios.

## Project Structure

```
RL-Resource-Optimizer/
├── main.py                         # CLI entry point: train, evaluate, compare
├── config/
│   └── default.yaml                # Hyperparameters and environment settings
├── src/
│   ├── environment.py              # Custom Gymnasium environment
│   ├── reward.py                   # Composite reward function
│   ├── training.py                 # Training loop with logging and checkpoints
│   ├── evaluation.py               # Evaluation, comparison tables, and charts
│   └── agents/
│       ├── dqn_agent.py            # DQN with replay buffer and target network
│       ├── ppo_agent.py            # PPO with actor-critic and GAE
│       └── baseline_schedulers.py  # Round-robin, greedy, random baselines
├── results/                        # Checkpoints, logs, and plots (generated)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

### Train agents

```bash
# Train both DQN and PPO with default settings
python main.py train --agent both

# Train only DQN for 300 episodes
python main.py train --agent dqn --episodes 300

# Train only PPO
python main.py train --agent ppo --episodes 500
```

Training saves checkpoints to `results/checkpoints/`, episode logs to `results/`, and a training curve plot.

### Evaluate a checkpoint

```bash
python main.py evaluate --agent dqn --checkpoint results/checkpoints/dqn_best.pt
python main.py evaluate --agent ppo
```

### Compare all agents

```bash
python main.py compare
```

This runs DQN, PPO, and all three baselines over evaluation episodes with fixed seeds, then:
- Prints a summary table (mean/std for reward, throughput, latency, fairness)
- Runs an adaptability analysis under increasing node failure rates
- Saves CSV results and PNG charts to `results/`

## Environment

`ResourceAllocationEnv` is a custom Gymnasium environment where:

- **State**: Per-node CPU/memory utilization and availability, active status; head-of-queue task demands; queue length.
- **Action**: Assign the next queued task to one of N nodes.
- **Dynamics**: Stochastic task arrivals, node failures/recoveries, capacity fluctuations.
- **Reward**: Weighted combination of throughput, latency, fairness, and rejection penalties (configurable in `config/default.yaml`).

## Agents

| Agent | Description |
|-------|-------------|
| **DQN** | Deep Q-Network with experience replay, target network, epsilon-greedy exploration |
| **PPO** | Proximal Policy Optimization with shared actor-critic trunk, GAE, clipped surrogate loss |
| **Round-Robin** | Cyclic assignment across active nodes |
| **Greedy** | Assigns to the node with the most available CPU |
| **Random** | Uniform random assignment to active nodes |

## Configuration

All hyperparameters and environment settings are in `config/default.yaml`. Key sections:

- `environment`: Node count, capacities, task arrival distribution, failure/recovery rates
- `reward`: Weights for throughput, latency, and fairness components
- `dqn` / `ppo`: Learning rates, network architecture, exploration parameters
- `evaluation`: Number of eval episodes, random seed

## Tech Stack

- **Python** -- core language
- **PyTorch** -- neural network training for DQN and PPO
- **Gymnasium** -- environment interface
- **NumPy** -- numerical computation
- **Pandas** -- metrics aggregation and reporting
- **Matplotlib** -- visualization

