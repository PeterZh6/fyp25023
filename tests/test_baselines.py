"""Tests for baseline policies."""

import numpy as np
import pytest

from rl.budget_env import EnvConfig, AnalysisBudgetEnv
from rl.baselines import (
    all_skip, all_l1, all_l2, all_l3,
    greedy_cheap, budget_aware, escalation,
    evaluate_policy,
)


@pytest.fixture
def all_easy_env():
    cfg = EnvConfig(
        num_sites=50,
        budget=200.0,
        difficulty_distribution=[1.0, 0.0, 0.0],
        success_rates={
            "SKIP": [0, 0, 0],
            "L1": [0.9, 0.5, 0.1],
            "L2": [1.0, 0.8, 0.3],
            "L3": [1.0, 0.95, 0.7],
        },
    )
    return AnalysisBudgetEnv(cfg)


@pytest.fixture
def all_hard_env():
    cfg = EnvConfig(
        num_sites=20,
        budget=500.0,
        difficulty_distribution=[0.0, 0.0, 1.0],
        success_rates={
            "SKIP": [0, 0, 0],
            "L1": [1.0, 0.5, 0.0],
            "L2": [1.0, 0.8, 0.2],
            "L3": [1.0, 0.95, 0.8],
        },
    )
    return AnalysisBudgetEnv(cfg)


def test_all_skip_resolves_nothing(all_easy_env):
    metrics = evaluate_policy(all_skip, all_easy_env, n_episodes=50, seed=0)
    assert metrics["mean_resolved"] == 0.0
    assert metrics["mean_cost"] == 0.0


def test_all_l1_on_easy(all_easy_env):
    metrics = evaluate_policy(all_l1, all_easy_env, n_episodes=100, seed=0)
    assert metrics["mean_resolve_rate"] > 0.8


def test_escalation_on_hard(all_hard_env):
    """Escalation should try L1 -> L2 -> L3 on hard targets."""
    env = all_hard_env
    obs, _ = env.reset(seed=42)

    actions = []
    for _ in range(100):
        a = escalation(obs, env)
        actions.append(a)
        obs, _, done, _, _ = env.step(a)
        if done:
            break

    assert 1 in actions
    assert 2 in actions
    assert 3 in actions


def test_greedy_cheap_prefers_l1(all_easy_env):
    env = all_easy_env
    obs, _ = env.reset(seed=0)

    a = greedy_cheap(obs, env)
    assert a == 1


def test_budget_aware_adapts():
    """With lots of budget, budget_aware should pick expensive levels."""
    cfg = EnvConfig(
        num_sites=5, budget=10000.0,
        difficulty_distribution=[1.0, 0.0, 0.0],
    )
    env = AnalysisBudgetEnv(cfg)
    obs, _ = env.reset(seed=0)

    a = budget_aware(obs, env)
    assert a == 3


def test_all_l2_fallback():
    """All-L2 should fallback to L1 when L2 budget insufficient."""
    cfg = EnvConfig(num_sites=10, budget=3.0)
    env = AnalysisBudgetEnv(cfg)
    obs, _ = env.reset(seed=0)

    a = all_l2(obs, env)
    assert a == 1


def test_evaluate_policy_returns_correct_keys():
    cfg = EnvConfig(num_sites=10, budget=50.0)
    env = AnalysisBudgetEnv(cfg)
    metrics = evaluate_policy(all_l1, env, n_episodes=10, seed=0)

    expected_keys = [
        "mean_reward", "std_reward",
        "mean_cost", "std_cost",
        "mean_resolved", "std_resolved",
        "mean_resolve_rate", "std_resolve_rate",
        "mean_episode_length",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing key: {key}"
