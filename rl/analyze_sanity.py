"""
Sanity check analysis script for the calibrated environment.
Runs quick validation of env, baselines, and optionally a short RL training.

Usage:
    python -m rl.analyze_sanity [--train]
"""

import argparse
import os

import numpy as np

from rl.budget_env import EnvConfig, AnalysisBudgetEnv
from rl.baselines import ALL_BASELINES, evaluate_policy, run_all_baselines


def check_env_basics(binary: str = "gcc"):
    """Verify env creates, resets, steps correctly."""
    print("=" * 60)
    print(f"1. ENV BASICS ({binary})")
    print("=" * 60)

    cfg = EnvConfig.from_yaml("rl/configs/env_configs.yaml", binary)
    env = AnalysisBudgetEnv(cfg)

    obs, info = env.reset(seed=42)
    from rl.budget_env import OBS_DIM
    assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},) got {obs.shape}"
    assert 0 <= obs[0] <= 1, "progress out of range"
    assert abs(obs[2] - 1.0) < 0.01, "remain_ratio should start ~1.0"
    print(f"  obs shape: {obs.shape} ✓")
    print(f"  initial obs[:5]: {obs[:5]}")

    total_r = 0
    for step in range(10000):
        obs, r, done, trunc, info = env.step(env.action_space.sample())
        total_r += r
        if done:
            break

    assert info["budget_spent"] <= cfg.budget + cfg.costs["L3"], "budget overrun"
    print(f"  Episode done in {step+1} steps ✓")
    print(f"  Resolved: {info['total_resolved']}/{cfg.num_sites}")
    print(f"  Budget spent: {info['budget_spent']:.1f}/{cfg.budget:.1f}")
    print(f"  Total reward: {total_r:.2f}")


def check_skip_cost():
    """SKIP should cost nothing."""
    print("\n" + "=" * 60)
    print("2. SKIP COST CHECK")
    print("=" * 60)

    cfg = EnvConfig(num_sites=5, budget=100.0)
    env = AnalysisBudgetEnv(cfg)
    env.reset(seed=0)

    _, _, _, _, info = env.step(0)
    assert info["budget_spent"] == 0.0, "SKIP should cost 0"
    print("  SKIP costs 0 ✓")


def check_repeat_penalty():
    """Repeating same level should trigger penalty."""
    print("\n" + "=" * 60)
    print("3. REPEAT PENALTY CHECK")
    print("=" * 60)

    cfg = EnvConfig(num_sites=10, budget=1000.0,
                    success_rates={"SKIP": [0, 0, 0], "L1": [0, 0, 0],
                                   "L2": [0, 0, 0], "L3": [0, 0, 0]})
    env = AnalysisBudgetEnv(cfg)
    env.reset(seed=0)

    _, r1, _, _, _ = env.step(1)
    _, r2, _, _, _ = env.step(1)

    expected_penalty = cfg.invalid_repeat_penalty - cfg.cost_lambda * cfg.costs["L1"]
    assert abs(r2 - expected_penalty) < 1e-6, f"Expected {expected_penalty}, got {r2}"
    print(f"  First L1: reward={r1:.4f}")
    print(f"  Repeat L1: reward={r2:.4f} (includes penalty) ✓")


def check_oracle_mode():
    """Oracle obs should be one-hot for difficulty."""
    print("\n" + "=" * 60)
    print("4. ORACLE MODE CHECK")
    print("=" * 60)

    cfg = EnvConfig(num_sites=20, budget=200.0, oracle_mode=True)
    env = AnalysisBudgetEnv(cfg)
    obs, _ = env.reset(seed=42)

    oracle_bits = obs[16:19]
    assert oracle_bits.sum() == 1.0, f"Oracle should be one-hot, got {oracle_bits}"
    print(f"  Oracle bits: {oracle_bits} ✓")


def check_baselines(binary: str = "gcc"):
    """Run baselines and verify reasonable results."""
    print("\n" + "=" * 60)
    print(f"5. BASELINES ({binary})")
    print("=" * 60)

    cfg = EnvConfig.from_yaml("rl/configs/env_configs.yaml", binary)
    run_all_baselines(cfg, n_episodes=200, seed=42)


def check_short_training(binary: str = "gcc"):
    """Quick training to verify RL pipeline works."""
    print("\n" + "=" * 60)
    print(f"6. SHORT RL TRAINING ({binary}, 20k steps)")
    print("=" * 60)

    from rl.train import TrainConfig, train_single, evaluate_trained_model
    import dataclasses

    cfg = TrainConfig(
        binary_name=binary,
        total_timesteps=20_000,
        save_dir="results/sanity_check",
    )
    result = train_single(cfg, seed=0)
    eval_result = evaluate_trained_model(result["best_model_path"], cfg, n_episodes=100)

    print(f"  RL resolve_rate: {eval_result['rl']['mean_resolve_rate']:.3f}")
    print(f"  Best baseline: {eval_result['best_baseline']} "
          f"({eval_result['baselines'][eval_result['best_baseline']]['mean_resolve_rate']:.3f})")
    print(f"  Improvement: {eval_result['improvement_over_best']:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="gcc")
    parser.add_argument("--train", action="store_true",
                        help="Also run a short RL training check")
    args = parser.parse_args()

    check_env_basics(args.binary)
    check_skip_cost()
    check_repeat_penalty()
    check_oracle_mode()
    check_baselines(args.binary)

    if args.train:
        check_short_training(args.binary)

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
