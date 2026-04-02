"""Budgeted analysis-strategy selection as a Gymnasium MDP.

Environment with:
  - Hard budget constraint
  - Cluster-dependent hidden difficulty (easy/medium/hard)
  - Information gain via visible per-cluster historical stats
  - Per-target multi-step attempts
  - Optional SKIP action for budget planning
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    n_clusters: int = 4
    targets_per_cluster: int = 5
    budget: int = 60
    seed: int = 0

    # Action space = {0:SKIP, 1:L1, 2:L2, 3:L3}
    costs: Dict[int, int] = dataclasses.field(
        default_factory=lambda: {0: 0, 1: 1, 2: 5, 3: 20}
    )

    type_probs: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"easy": 0.2, "medium": 0.3, "hard": 0.5}
    )

    # Success probabilities per hidden type and action level
    success_probs: Dict[str, Dict[int, float]] = dataclasses.field(
        default_factory=lambda: {
            "easy":   {1: 0.80, 2: 0.95, 3: 1.00},
            "medium": {1: 0.40, 2: 0.80, 3: 0.95},
            "hard":   {1: 0.10, 2: 0.50, 3: 0.85},
        }
    )

    reward_solved: float = 1.0
    cost_lambda: float = 0.02
    overbudget_penalty: float = -1.0

    max_attempts_per_target: int = 3
    invalid_repeat_penalty: float = -0.05

    use_cluster_stats: bool = True
    oracle_mode: bool = False


class AnalysisBudgetEnv(gym.Env):
    """Budgeted analysis strategy selection with cluster-dependent hidden difficulty.

    Episode:
      - N targets arranged by clusters (cluster_id = idx // targets_per_cluster).
      - For each target, agent may choose L1/L2/L3 (up to max_attempts) or SKIP.
      - Budget is a hard constraint: cost > remaining => terminate.

    Observation vector (float32):
      [progress, one_hot(cluster_id), spent_ratio, remain_ratio,
       tried_mask(3), cluster_attempts_norm(3), cluster_success_rate(3),
       global_success_rate, oracle_type_oh(3)]
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg

        self.n_clusters = cfg.n_clusters
        self.targets_per_cluster = cfg.targets_per_cluster
        self.n_targets = self.n_clusters * self.targets_per_cluster

        self.action_space = spaces.Discrete(4)

        obs_dim = self.n_clusters + 16
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._np_rng = np.random.default_rng(cfg.seed)

        self.cluster_types: List[str] = []
        self.spent: int = 0
        self.cur_target: int = 0
        self.total_solved: int = 0

        self.cluster_attempts = np.zeros((self.n_clusters, 3), dtype=np.int32)
        self.cluster_successes = np.zeros((self.n_clusters, 3), dtype=np.int32)

        self.target_tried_mask = np.zeros(3, dtype=np.int32)
        self.target_attempts = 0
        self._episode_cost = 0

    def _sample_cluster_types(self) -> List[str]:
        types = list(self.cfg.type_probs.keys())
        probs = np.array([self.cfg.type_probs[t] for t in types], dtype=np.float64)
        probs = probs / probs.sum()
        sampled = self._np_rng.choice(types, size=self.n_clusters, replace=True, p=probs)
        return list(sampled)

    def _cluster_id(self, target_idx: int) -> int:
        return int(target_idx // self.targets_per_cluster)

    def _one_hot(self, idx: int, size: int) -> np.ndarray:
        v = np.zeros(size, dtype=np.float32)
        v[idx] = 1.0
        return v

    def _get_obs(self) -> np.ndarray:
        progress = np.array([self.cur_target / max(1, self.n_targets)], dtype=np.float32)
        cid = self._cluster_id(self.cur_target) if self.cur_target < self.n_targets else 0
        cid_oh = self._one_hot(cid, self.n_clusters)

        spent_ratio = np.array([self.spent / max(1, self.cfg.budget)], dtype=np.float32)
        remain_ratio = np.array([
            (self.cfg.budget - self.spent) / max(1, self.cfg.budget)
        ], dtype=np.float32)

        tried_mask = self.target_tried_mask.astype(np.float32)

        attempts = self.cluster_attempts[cid].astype(np.float32)
        denom = max(1.0, float(attempts.sum()))

        if self.cfg.use_cluster_stats:
            attempts_norm = attempts / denom
            successes = self.cluster_successes[cid].astype(np.float32)
            success_rate = successes / (attempts + 1e-6)
        else:
            attempts_norm = np.zeros(3, dtype=np.float32)
            success_rate = np.zeros(3, dtype=np.float32)

        global_success_rate = np.array([
            self.total_solved / max(1, self.cur_target)
        ], dtype=np.float32)

        if self.cfg.oracle_mode:
            type_map = {"easy": 0, "medium": 1, "hard": 2}
            hidden_type = self.cluster_types[cid]
            type_oh = self._one_hot(type_map[hidden_type], 3)
        else:
            type_oh = np.zeros(3, dtype=np.float32)

        obs = np.concatenate([
            progress, cid_oh, spent_ratio, remain_ratio,
            tried_mask, attempts_norm, success_rate,
            global_success_rate, type_oh,
        ]).astype(np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        self.cluster_types = self._sample_cluster_types()
        self.spent = 0
        self.cur_target = 0
        self.total_solved = 0
        self.cluster_attempts[:] = 0
        self.cluster_successes[:] = 0

        self.target_tried_mask[:] = 0
        self.target_attempts = 0
        self._episode_cost = 0

        obs = self._get_obs()
        info = {
            "budget": self.cfg.budget,
            "cluster_types": self.cluster_types,
        }
        return obs, info

    def _advance_target(self):
        self.cur_target += 1
        self.target_tried_mask[:] = 0
        self.target_attempts = 0

    def step(self, action: int):
        assert self.action_space.contains(action)

        if self.cur_target >= self.n_targets:
            return self._get_obs(), 0.0, True, False, {"reason": "done"}

        cost = self.cfg.costs[int(action)]
        remaining = self.cfg.budget - self.spent

        if cost > remaining:
            terminated = True
            reward = self.cfg.overbudget_penalty
            info = {
                "reason": "overbudget",
                "spent": self.spent,
                "remaining": remaining,
                "action_cost": cost,
                "solved": self.total_solved,
            }
            return self._get_obs(), float(reward), terminated, False, info

        self.spent += cost
        self._episode_cost += cost

        cid = self._cluster_id(self.cur_target)
        hidden_type = self.cluster_types[cid]

        reward = 0.0
        solved_now = False
        invalid_repeat = False

        if action == 0:
            self._advance_target()
        else:
            level = int(action)
            mask_idx = level - 1
            if self.target_tried_mask[mask_idx] == 1:
                invalid_repeat = True
            self.target_tried_mask[mask_idx] = 1
            self.target_attempts += 1

            self.cluster_attempts[cid, mask_idx] += 1

            p = self.cfg.success_probs[hidden_type][level]
            if float(self._np_rng.random()) < p:
                solved_now = True
                self.total_solved += 1
                self.cluster_successes[cid, mask_idx] += 1
                self._advance_target()
            else:
                if self.target_attempts >= self.cfg.max_attempts_per_target:
                    self._advance_target()

        if solved_now:
            reward += self.cfg.reward_solved
        reward -= self.cfg.cost_lambda * cost
        if invalid_repeat:
            reward += self.cfg.invalid_repeat_penalty

        terminated = self.cur_target >= self.n_targets

        info = {
            "spent": self.spent,
            "remaining": self.cfg.budget - self.spent,
            "episode_cost": self._episode_cost,
            "solved": self.total_solved,
            "cur_target": self.cur_target,
            "action": int(action),
            "solved_now": solved_now,
            "invalid_repeat": invalid_repeat,
            "cluster_id": cid,
            "cluster_type": hidden_type,
        }

        return self._get_obs(), float(reward), terminated, False, info
