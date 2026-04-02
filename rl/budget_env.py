"""Budgeted analysis-strategy selection as a Gymnasium MDP.

Environment with:
  - Hard budget constraint
  - Per-site independent difficulty sampling (easy/medium/hard)
  - Information gain via global historical stats + per-site features
  - Per-target multi-step attempts (up to 3, no repeat of same level)
  - Optional SKIP action for budget planning
  - Loads calibrated parameters from YAML config
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces

OBS_DIM = 19
LEVEL_NAMES = ["SKIP", "L1", "L2", "L3"]
DIFFICULTY_NAMES = ["easy", "medium", "hard"]


@dataclass
class EnvConfig:
    num_sites: int = 100
    budget: float = 60.0
    costs: Dict[str, float] = field(
        default_factory=lambda: {"SKIP": 0, "L1": 1.0, "L2": 5.0, "L3": 20.0}
    )

    difficulty_distribution: List[float] = field(
        default_factory=lambda: [0.33, 0.34, 0.33]
    )

    success_rates: Dict[str, List[float]] = field(default_factory=lambda: {
        "SKIP": [0.0, 0.0, 0.0],
        "L1":   [0.9, 0.5, 0.1],
        "L2":   [1.0, 0.8, 0.3],
        "L3":   [1.0, 0.95, 0.7],
    })

    jump_ratio: float = 0.5
    type_agree_ratio: float = 0.8

    reward_solved: float = 1.0
    cost_lambda: float = 0.02
    overbudget_penalty: float = -1.0
    invalid_repeat_penalty: float = -0.05
    max_attempts_per_target: int = 3

    oracle_mode: bool = False
    use_global_stats: bool = True
    use_site_features: bool = True
    budget_ratio: float = 2.0
    max_sites_per_episode: int = 2000

    @classmethod
    def from_yaml(cls, yaml_path: str, binary_name: str = "mixed_no_cpp") -> "EnvConfig":
        """Load calibrated parameters from YAML config.

        Supports exact match, top-level key match, and substring match
        (if unique) for binary_name.
        """
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        src = None
        per_binary = cfg.get("per_binary", {})
        top_level_keys = [
            k for k in cfg if k not in ("defaults", "per_binary", "estimated_fields")
        ]

        if binary_name in per_binary:
            src = per_binary[binary_name]
        elif binary_name in cfg:
            src = cfg[binary_name]
        else:
            def _short_name(full: str) -> str:
                """Extract short name: 'gcc_base.arm32-gcc81-O3' -> 'gcc'."""
                return full.split("_base.")[0] if "_base." in full else full

            short_matches = [
                k for k in per_binary
                if _short_name(k).lower() == binary_name.lower()
            ]
            if len(short_matches) == 1:
                src = per_binary[short_matches[0]]
            elif len(short_matches) > 1:
                raise ValueError(
                    f"Ambiguous short name '{binary_name}', matches: {short_matches}"
                )
            else:
                top_matches = [
                    k for k in top_level_keys
                    if binary_name.lower() in k.lower()
                ]
                if len(top_matches) == 1:
                    src = cfg[top_matches[0]]
                else:
                    sub_matches = [
                        k for k in per_binary
                        if binary_name.lower() in k.lower()
                    ]
                    if len(sub_matches) == 1:
                        src = per_binary[sub_matches[0]]
                    elif len(sub_matches) > 1:
                        raise ValueError(
                            f"Ambiguous binary name '{binary_name}', matches: {sub_matches}"
                        )
                    else:
                        available = list(per_binary.keys()) + top_level_keys
                        raise ValueError(
                            f"Binary '{binary_name}' not found. Available: {available}"
                        )

        costs_raw = cfg.get("defaults", {}).get("costs", {"SKIP": 0, "L1": 1, "L2": 5, "L3": 20})
        costs = {k: float(v) for k, v in costs_raw.items()}

        num_sites = src["num_sites"]

        instance = cls(
            num_sites=num_sites,
            costs=costs,
            difficulty_distribution=src["difficulty_distribution"],
            success_rates={
                k: [float(x) for x in v]
                for k, v in src["success_rates"].items()
            },
            jump_ratio=src.get("jump_ratio", 0.5),
            type_agree_ratio=src.get("type_agree_ratio", 0.8),
        )
        instance.num_sites = min(num_sites, instance.max_sites_per_episode)
        instance.budget = instance.budget_ratio * instance.num_sites * costs["L1"]
        return instance


class AnalysisBudgetEnv(gym.Env):
    """Budgeted analysis strategy selection with per-site independent difficulty.

    Episode:
      - N sites, each independently sampled difficulty and type.
      - For each site, agent may choose L1/L2/L3 (up to max_attempts) or SKIP.
      - Budget is a hard constraint: cost > remaining => terminate.

    Observation vector (float32, dim=18):
      [progress, spent_ratio, remain_ratio,
       tried_L1, tried_L2, tried_L3,
       attempts_norm, current_resolved,
       global_success_rate, global_L1_sr, global_L2_sr, global_L3_sr,
       site_is_jump, binary_easy_ratio, binary_medium_ratio, binary_hard_ratio,
       oracle_easy, oracle_medium, oracle_hard]
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.site_difficulties: np.ndarray = np.array([])
        self.site_types: np.ndarray = np.array([])
        self.current_target: int = 0
        self.budget_remaining: float = 0.0
        self.budget_spent: float = 0.0
        self.resolved: np.ndarray = np.array([])
        self.attempts: np.ndarray = np.array([])
        self.tried_levels: np.ndarray = np.array([])

        self.total_attempts_by_level = {"L1": 0, "L2": 0, "L3": 0}
        self.total_successes_by_level = {"L1": 0, "L2": 0, "L3": 0}
        self.total_resolved = 0
        self.total_visited = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        dist = self.config.difficulty_distribution
        dist_arr = np.array(dist, dtype=np.float64)
        dist_arr = dist_arr / dist_arr.sum()

        self.site_difficulties = self.np_random.choice(
            DIFFICULTY_NAMES,
            size=self.config.num_sites,
            p=dist_arr,
        )

        self.site_types = self.np_random.choice(
            ["jump", "call"],
            size=self.config.num_sites,
            p=[self.config.jump_ratio, 1 - self.config.jump_ratio],
        )

        self.current_target = 0
        self.budget_remaining = self.config.budget
        self.budget_spent = 0.0
        self.resolved = np.zeros(self.config.num_sites, dtype=bool)
        self.attempts = np.zeros(self.config.num_sites, dtype=int)
        self.tried_levels = np.zeros((self.config.num_sites, 3), dtype=bool)

        self.total_attempts_by_level = {"L1": 0, "L2": 0, "L3": 0}
        self.total_successes_by_level = {"L1": 0, "L2": 0, "L3": 0}
        self.total_resolved = 0
        self.total_visited = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action)

        if self.current_target >= self.config.num_sites:
            return self._get_obs(), 0.0, True, False, self._get_info()

        level = LEVEL_NAMES[action]
        cost = self.config.costs[level]

        reward = 0.0
        terminated = False
        truncated = False

        if cost > self.budget_remaining:
            reward = self.config.overbudget_penalty
            terminated = True
            return self._get_obs(), float(reward), terminated, truncated, self._get_info()

        self.budget_remaining -= cost
        self.budget_spent += cost
        reward -= self.config.cost_lambda * cost

        if level == "SKIP":
            self._advance_target()
        else:
            level_idx = ["L1", "L2", "L3"].index(level)

            if self.tried_levels[self.current_target, level_idx]:
                reward += self.config.invalid_repeat_penalty
            else:
                self.tried_levels[self.current_target, level_idx] = True

            self.attempts[self.current_target] += 1
            self.total_attempts_by_level[level] += 1

            difficulty = self.site_difficulties[self.current_target]
            diff_idx = DIFFICULTY_NAMES.index(difficulty)
            success_prob = self.config.success_rates[level][diff_idx]
            success = self.np_random.random() < success_prob

            if success and not self.resolved[self.current_target]:
                self.resolved[self.current_target] = True
                self.total_resolved += 1
                self.total_successes_by_level[level] += 1
                reward += self.config.reward_solved

            if (self.resolved[self.current_target] or
                    self.attempts[self.current_target] >= self.config.max_attempts_per_target):
                self._advance_target()

        if self.current_target >= self.config.num_sites:
            terminated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _advance_target(self):
        self.total_visited += 1
        self.current_target += 1

    def _get_obs(self) -> np.ndarray:
        if self.current_target >= self.config.num_sites:
            return np.zeros(OBS_DIM, dtype=np.float32)

        ct = self.current_target
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0] = ct / self.config.num_sites
        obs[1] = self.budget_spent / max(self.config.budget, 1e-6)
        obs[2] = self.budget_remaining / max(self.config.budget, 1e-6)
        obs[3:6] = self.tried_levels[ct].astype(np.float32)
        obs[6] = self.attempts[ct] / self.config.max_attempts_per_target
        obs[7] = float(self.resolved[ct])

        if self.config.use_global_stats:
            visited = max(self.total_visited, 1)
            obs[8] = self.total_resolved / visited
            for i, lvl in enumerate(["L1", "L2", "L3"]):
                att = self.total_attempts_by_level[lvl]
                obs[9 + i] = self.total_successes_by_level[lvl] / max(att, 1)

        if self.config.use_site_features:
            obs[12] = 1.0 if self.site_types[ct] == "jump" else 0.0
            obs[13] = self.config.difficulty_distribution[0]
            obs[14] = self.config.difficulty_distribution[1]
            obs[15] = self.config.difficulty_distribution[2]

        if self.config.oracle_mode:
            diff = self.site_difficulties[ct]
            oracle_idx = DIFFICULTY_NAMES.index(diff)
            obs[16 + oracle_idx] = 1.0

        return obs

    def _get_info(self) -> dict:
        return {
            "budget": self.config.budget,
            "budget_spent": self.budget_spent,
            "budget_remaining": self.budget_remaining,
            "current_target": self.current_target,
            "total_resolved": self.total_resolved,
            "total_visited": self.total_visited,
            "num_sites": self.config.num_sites,
            "resolve_rate": (
                self.total_resolved / max(self.total_visited, 1)
            ),
        }
