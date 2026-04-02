"""Baseline (hand-crafted) policies and evaluation harness for the budgeted analysis environment."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rl.budget_env import AnalysisBudgetEnv, EnvConfig

PolicyFn = Callable[[np.ndarray, AnalysisBudgetEnv], int]


def all_skip(obs: np.ndarray, env: AnalysisBudgetEnv) -> int:
    return 0


def all_l1(obs: np.ndarray, env: AnalysisBudgetEnv) -> int:
    return 1


def all_l2(obs: np.ndarray, env: AnalysisBudgetEnv) -> int:
    costs = env.config.costs
    if env.budget_remaining >= costs["L2"]:
        return 2
    if env.budget_remaining >= costs["L1"]:
        return 1
    return 0


def all_l3(obs: np.ndarray, env: AnalysisBudgetEnv) -> int:
    costs = env.config.costs
    if env.budget_remaining >= costs["L3"]:
        return 3
    if env.budget_remaining >= costs["L2"]:
        return 2
    if env.budget_remaining >= costs["L1"]:
        return 1
    return 0


def random_policy(obs: np.ndarray, env: AnalysisBudgetEnv) -> int:
    return int(env.np_random.integers(0, 4))


def greedy_cheap(obs: np.ndarray, env: AnalysisBudgetEnv) -> int:
    """Pick cheapest untried non-SKIP level."""
    ct = env.current_target
    if ct >= env.config.num_sites:
        return 0
    costs = env.config.costs
    for lvl_idx, lvl_name in enumerate(["L1", "L2", "L3"]):
        if not env.tried_levels[ct, lvl_idx] and env.budget_remaining >= costs[lvl_name]:
            return lvl_idx + 1
    return 0


def budget_aware(obs: np.ndarray, env: AnalysisBudgetEnv) -> int:
    """Dynamically adjust action based on remaining budget per site."""
    ct = env.current_target
    if ct >= env.config.num_sites:
        return 0
    remaining_sites = max(env.config.num_sites - ct, 1)
    avg_budget = env.budget_remaining / remaining_sites
    costs = env.config.costs

    for lvl_idx, lvl_name in reversed(list(enumerate(["L1", "L2", "L3"]))):
        if (avg_budget >= costs[lvl_name]
                and not env.tried_levels[ct, lvl_idx]
                and env.budget_remaining >= costs[lvl_name]):
            return lvl_idx + 1
    return 0


def escalation(obs: np.ndarray, env: AnalysisBudgetEnv) -> int:
    """Mimics human analyst: try L1 first, escalate to L2, then L3."""
    ct = env.current_target
    if ct >= env.config.num_sites:
        return 0
    if env.resolved[ct]:
        return 0
    costs = env.config.costs
    for lvl_idx, lvl_name in enumerate(["L1", "L2", "L3"]):
        if not env.tried_levels[ct, lvl_idx] and env.budget_remaining >= costs[lvl_name]:
            return lvl_idx + 1
    return 0


ALL_BASELINES: Dict[str, PolicyFn] = {
    "all_skip": all_skip,
    "all_l1": all_l1,
    "all_l2": all_l2,
    "all_l3": all_l3,
    "random": random_policy,
    "greedy_cheap": greedy_cheap,
    "budget_aware": budget_aware,
    "escalation": escalation,
}


def evaluate_policy(
    policy_fn: PolicyFn,
    env: AnalysisBudgetEnv,
    n_episodes: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate a policy function on an environment.

    Returns dict with mean/std for reward, cost, resolved, resolve_rate,
    episode_length, plus per_episode details.
    """
    rewards: List[float] = []
    costs: List[float] = []
    resolved_counts: List[int] = []
    resolve_rates: List[float] = []
    episode_lengths: List[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0

        done = False
        while not done:
            action = policy_fn(obs, env)
            obs, r, terminated, truncated, info = env.step(action)
            total_reward += r
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        costs.append(info["budget_spent"])
        resolved_counts.append(info["total_resolved"])
        resolve_rates.append(info["resolve_rate"])
        episode_lengths.append(steps)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "mean_resolved": float(np.mean(resolved_counts)),
        "std_resolved": float(np.std(resolved_counts)),
        "mean_resolve_rate": float(np.mean(resolve_rates)),
        "std_resolve_rate": float(np.std(resolve_rates)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }


def run_all_baselines(
    env_config: EnvConfig,
    n_episodes: int = 1000,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run all baselines on a given env config and optionally save results."""
    env = AnalysisBudgetEnv(env_config)
    results: Dict[str, Dict[str, Any]] = {}

    print(f"\n{'Baseline':<16s} {'Reward':>10s} {'Cost':>10s} {'Resolved':>10s} "
          f"{'Rate':>8s} {'Steps':>8s}")
    print("-" * 70)

    for name, policy_fn in ALL_BASELINES.items():
        metrics = evaluate_policy(policy_fn, env, n_episodes=n_episodes, seed=seed)
        results[name] = metrics
        print(
            f"{name:<16s} "
            f"{metrics['mean_reward']:>9.2f} "
            f"{metrics['mean_cost']:>9.1f} "
            f"{metrics['mean_resolved']:>9.1f} "
            f"{metrics['mean_resolve_rate']:>7.3f} "
            f"{metrics['mean_episode_length']:>7.0f}"
        )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = {
            "num_sites": env_config.num_sites,
            "budget": env_config.budget,
            "budget_ratio": env_config.budget_ratio,
            "n_episodes": n_episodes,
            "baselines": results,
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to: {output_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--config", default="rl/configs/env_configs.yaml")
    parser.add_argument("--binary", default="gcc")
    parser.add_argument("--budget-ratio", type=float, default=2.0)
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    env_config = EnvConfig.from_yaml(args.config, args.binary)
    env_config.budget_ratio = args.budget_ratio
    env_config.budget = env_config.budget_ratio * env_config.num_sites * env_config.costs["L1"]

    if args.output is None:
        args.output = f"data/calibration/baseline_results_{args.binary}.json"

    run_all_baselines(env_config, n_episodes=args.n_episodes, seed=args.seed,
                      output_path=args.output)


if __name__ == "__main__":
    main()
