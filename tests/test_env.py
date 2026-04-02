"""Tests for AnalysisBudgetEnv."""

import os

import numpy as np
import pytest

from rl.budget_env import EnvConfig, AnalysisBudgetEnv, OBS_DIM

assert OBS_DIM == 19, f"Expected OBS_DIM=19, got {OBS_DIM}"


@pytest.fixture
def default_env():
    cfg = EnvConfig(num_sites=20, budget=100.0)
    return AnalysisBudgetEnv(cfg)


@pytest.fixture
def yaml_env():
    yaml_path = "rl/configs/env_configs.yaml"
    if not os.path.exists(yaml_path):
        pytest.skip("YAML config not available")
    cfg = EnvConfig.from_yaml(yaml_path, "gcc")
    return AnalysisBudgetEnv(cfg)


class TestObsShape:
    def test_default_obs_shape(self, default_env):
        obs, _ = default_env.reset(seed=42)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_yaml_obs_shape(self, yaml_env):
        obs, _ = yaml_env.reset(seed=42)
        assert obs.shape == (OBS_DIM,)

    def test_obs_bounds(self, default_env):
        obs, _ = default_env.reset(seed=0)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)


class TestBudgetConstraint:
    def test_budget_not_exceeded(self, default_env):
        default_env.reset(seed=0)
        total_cost = 0.0
        for _ in range(5000):
            obs, r, done, trunc, info = default_env.step(
                default_env.action_space.sample()
            )
            if done:
                break
        assert info["budget_spent"] <= default_env.config.budget + default_env.config.costs["L3"]

    def test_skip_costs_nothing(self, default_env):
        default_env.reset(seed=0)
        _, _, _, _, info = default_env.step(0)
        assert info["budget_spent"] == 0.0

    def test_overbudget_terminates(self):
        cfg = EnvConfig(num_sites=5, budget=0.5, costs={"SKIP": 0, "L1": 1.0, "L2": 5.0, "L3": 20.0})
        env = AnalysisBudgetEnv(cfg)
        env.reset(seed=0)
        _, r, done, _, _ = env.step(1)
        assert done
        assert r == cfg.overbudget_penalty


class TestRepeatPenalty:
    def test_repeat_same_level(self):
        cfg = EnvConfig(
            num_sites=10, budget=1000.0,
            success_rates={
                "SKIP": [0, 0, 0], "L1": [0, 0, 0],
                "L2": [0, 0, 0], "L3": [0, 0, 0],
            },
        )
        env = AnalysisBudgetEnv(cfg)
        env.reset(seed=0)

        _, r1, _, _, _ = env.step(1)
        _, r2, _, _, _ = env.step(1)

        cost_penalty = cfg.cost_lambda * cfg.costs["L1"]
        assert abs(r2 - (cfg.invalid_repeat_penalty - cost_penalty)) < 1e-6


class TestOracleMode:
    def test_oracle_one_hot(self):
        cfg = EnvConfig(num_sites=50, budget=500.0, oracle_mode=True)
        env = AnalysisBudgetEnv(cfg)
        obs, _ = env.reset(seed=42)

        oracle_bits = obs[16:19]
        assert oracle_bits.sum() == 1.0
        assert oracle_bits.max() == 1.0

    def test_no_oracle_zeros(self):
        cfg = EnvConfig(num_sites=50, budget=500.0, oracle_mode=False)
        env = AnalysisBudgetEnv(cfg)
        obs, _ = env.reset(seed=42)

        oracle_bits = obs[16:19]
        assert oracle_bits.sum() == 0.0


class TestFromYaml:
    def test_from_yaml_gcc(self):
        yaml_path = "rl/configs/env_configs.yaml"
        if not os.path.exists(yaml_path):
            pytest.skip("YAML config not available")
        cfg = EnvConfig.from_yaml(yaml_path, "gcc")
        assert cfg.num_sites == 1820
        assert abs(cfg.difficulty_distribution[0] - 0.5115) < 0.01

    def test_from_yaml_mixed(self):
        yaml_path = "rl/configs/env_configs.yaml"
        if not os.path.exists(yaml_path):
            pytest.skip("YAML config not available")
        cfg = EnvConfig.from_yaml(yaml_path, "mixed_no_cpp")
        assert cfg.num_sites == 2000

    def test_from_yaml_short_name(self):
        yaml_path = "rl/configs/env_configs.yaml"
        if not os.path.exists(yaml_path):
            pytest.skip("YAML config not available")
        cfg = EnvConfig.from_yaml(yaml_path, "openssl")
        assert cfg.num_sites == 82

    def test_from_yaml_invalid(self):
        yaml_path = "rl/configs/env_configs.yaml"
        if not os.path.exists(yaml_path):
            pytest.skip("YAML config not available")
        with pytest.raises(ValueError):
            EnvConfig.from_yaml(yaml_path, "nonexistent_binary_xyz")


class TestEpisodeCompletion:
    def test_all_skip_visits_all(self):
        cfg = EnvConfig(num_sites=10, budget=100.0)
        env = AnalysisBudgetEnv(cfg)
        env.reset(seed=0)

        for _ in range(10):
            _, _, done, _, info = env.step(0)
            if done:
                break

        assert info["total_visited"] == 10
        assert info["total_resolved"] == 0

    def test_success_increments_resolved(self):
        cfg = EnvConfig(
            num_sites=5, budget=100.0,
            difficulty_distribution=[1.0, 0.0, 0.0],
            success_rates={
                "SKIP": [0, 0, 0], "L1": [1.0, 0, 0],
                "L2": [1.0, 0, 0], "L3": [1.0, 0, 0],
            },
        )
        env = AnalysisBudgetEnv(cfg)
        env.reset(seed=42)

        _, r, _, _, info = env.step(1)
        assert r > 0
        assert info["total_resolved"] >= 1
