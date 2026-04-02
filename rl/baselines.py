"""Baseline (hand-crafted) policies for the budgeted analysis environment."""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np

from rl.budget_env import AnalysisBudgetEnv, EnvConfig


class BaselinePolicy:
    def reset(self):
        pass

    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError


class AlwaysL1(BaselinePolicy):
    def act(self, obs: np.ndarray) -> int:
        return 1


class AlwaysL3(BaselinePolicy):
    def act(self, obs: np.ndarray) -> int:
        return 3


class RandomPolicy(BaselinePolicy):
    def __init__(self, p_skip: float = 0.1, seed: int = 0):
        self.p_skip = p_skip
        self.rng = random.Random(seed)

    def act(self, obs: np.ndarray) -> int:
        if self.rng.random() < self.p_skip:
            return 0
        return self.rng.choice([1, 2, 3])


class GreedyFallback(BaselinePolicy):
    """Try L1 -> L2 -> L3 on each target; SKIP if all tried."""

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def act(self, obs: np.ndarray) -> int:
        idx = 1 + self.n_clusters + 1 + 1
        tried = obs[idx: idx + 3]
        tried = (tried > 0.5).astype(np.int32)

        if tried[0] == 0:
            return 1
        if tried[1] == 0:
            return 2
        if tried[2] == 0:
            return 3
        return 0


class BudgetAwareGreedy(BaselinePolicy):
    """Remaining-budget-aware fallback with cluster hardness heuristic."""

    def __init__(self, n_clusters: int, min_attempts_for_infer: int = 3):
        self.n_clusters = n_clusters
        self.min_attempts_for_infer = min_attempts_for_infer

    def act(self, obs: np.ndarray) -> int:
        progress_end = 1
        onehot_end = progress_end + self.n_clusters
        remain_idx = onehot_end + 1
        tried_idx = remain_idx + 1
        attempts_norm_idx = tried_idx + 3
        success_rate_idx = attempts_norm_idx + 3

        remain_ratio = float(obs[remain_idx])
        tried = (obs[tried_idx: tried_idx + 3] > 0.5).astype(np.int32)

        if tried.sum() > 0:
            if tried[0] == 0:
                return 1
            if tried[1] == 0:
                return 2
            if tried[2] == 0:
                return 3
            return 0

        l1_succ = float(obs[success_rate_idx + 0])

        if remain_ratio < 0.15:
            return 0
        if remain_ratio < 0.35:
            if l1_succ < 0.2:
                return 2
            return 1
        return 1


# ---- Rollout utilities ----

def run_episode(env: AnalysisBudgetEnv, policy: BaselinePolicy, seed: int = 0) -> Dict[str, Any]:
    obs, info0 = env.reset(seed=seed)
    policy.reset()

    total_reward = 0.0
    actions = []
    infos = []

    done = False
    while not done:
        a = policy.act(obs)
        obs, r, term, trunc, info = env.step(a)
        done = bool(term or trunc)
        total_reward += float(r)
        actions.append(int(a))
        infos.append(info)

    return {
        "return": total_reward,
        "solved": infos[-1].get("solved", 0) if infos else 0,
        "spent": infos[-1].get("spent", 0) if infos else 0,
        "actions": actions,
        "infos": infos,
        "cluster_types": info0.get("cluster_types"),
        "budget": info0.get("budget"),
    }


def eval_policy(env_cfg: EnvConfig, policy: BaselinePolicy, n_episodes: int = 200, seed0: int = 0) -> Dict[str, float]:
    env = AnalysisBudgetEnv(env_cfg)
    rets, solved, spent = [], [], []
    for i in range(n_episodes):
        out = run_episode(env, policy, seed=seed0 + i)
        rets.append(out["return"])
        solved.append(out["solved"])
        spent.append(out["spent"])

    return {
        "return_mean": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
        "solved_mean": float(np.mean(solved)),
        "solved_std": float(np.std(solved)),
        "spent_mean": float(np.mean(spent)),
        "spent_std": float(np.std(spent)),
    }
