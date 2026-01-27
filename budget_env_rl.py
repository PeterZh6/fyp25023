"""Budgeted analysis-strategy selection as an MDP (minimal but non-trivial).

What you get in this single file:
- A Gymnasium environment with:
  * hard budget constraint
  * cluster-dependent hidden difficulty (easy/medium/hard)
  * information gain via visible per-cluster historical stats
  * per-target multi-step attempts (so greedy fallback is meaningful)
  * optional SKIP action for budget planning
- Baseline policies: fixed, greedy fallback, budget-aware
- RL training/eval using stable-baselines3 (PPO or DQN)
- Plots:
  * solved vs budget (across budgets)
  * learning curve (mean episode return)
  * policy behavior: P(use expensive) vs remaining budget (rough)

Install deps (example):
  pip install gymnasium numpy matplotlib stable-baselines3

Run quick sanity:
  python budget_env_rl.py --mode sanity

Train PPO:
  python budget_env_rl.py --mode train --algo ppo --timesteps 200000

Evaluate across budgets (baselines + RL):
  python budget_env_rl.py --mode sweep --algo ppo --model out/model.zip

Notes:
- This is a synthetic environment; later you can replace the generator / features.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise RuntimeError(
        "Please install gymnasium: pip install gymnasium"
    ) from e


# -----------------------------
# Environment
# -----------------------------


@dataclass
class EnvConfig:
    n_clusters: int = 4
    targets_per_cluster: int = 5
    budget: int = 60
    seed: int = 0

    # Action levels and costs
    # Action space = {0:SKIP, 1:L1, 2:L2, 3:L3}
    costs: Dict[int, int] = dataclasses.field(
        default_factory=lambda: {0: 0, 1: 1, 2: 5, 3: 20}
    )

    # Per-cluster hidden type distribution
    type_probs: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"easy": 0.2, "medium": 0.3, "hard": 0.5}
    )

    # Success probabilities per hidden type and action level
    success_probs: Dict[str, Dict[int, float]] = dataclasses.field(
        default_factory=lambda: {
        "easy": {1: 0.95, 2: 0.99, 3: 1.00},   # O0/simple
        "medium": {1: 0.65, 2: 0.90, 3: 0.98}, # O1/O2
        "hard": {1: 0.20, 2: 0.60, 3: 0.90},   # O3/obfuscated
        }
    )

    # Reward shaping
    reward_solved: float = 1.0
    cost_lambda: float = 0.02
    overbudget_penalty: float = -1.0

    # Per-target attempt structure
    max_attempts_per_target: int = 3  # to prevent infinite loops
    invalid_repeat_penalty: float = -0.05  # if agent repeats same level on same target


class AnalysisBudgetEnv(gym.Env):
    """Budgeted analysis strategy selection with cluster-dependent hidden difficulty.

    Episode:
      - There are N targets arranged by clusters (cluster_id = idx // targets_per_cluster).
      - For each target, agent may choose a level (L1/L2/L3) multiple times (up to max_attempts),
        or SKIP.
      - If success: move to next target.
      - If failure: stays on same target and can choose again.
      - If SKIP: move to next target.
      - Budget is a hard constraint: if remaining budget < cost(action) => terminate.

    Partial observability:
      - Cluster hidden type is not observed.
      - Agent observes per-cluster attempt/success stats (information gain).

    Observation vector (float32):
      [
        progress (1),
        one_hot(cluster_id) (n_clusters),
        spent_ratio (1), remain_ratio (1),
        current_target_tried_mask (3),
        cluster_attempts_norm (3),
        cluster_success_rate (3),
        global_success_rate (1),
      ]
      Total dim = 1 + n_clusters + 1 + 1 + 3 + 3 + 3 + 1 = n_clusters + 13
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg

        self.n_clusters = cfg.n_clusters
        self.targets_per_cluster = cfg.targets_per_cluster
        self.n_targets = self.n_clusters * self.targets_per_cluster

        # Actions: 0=SKIP, 1=L1, 2=L2, 3=L3
        self.action_space = spaces.Discrete(4)

        obs_dim = self.n_clusters + 13
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # RNG
        self._np_rng = np.random.default_rng(cfg.seed)

        # Per-episode state
        self.cluster_types: List[str] = []
        self.spent: int = 0
        self.cur_target: int = 0
        self.total_solved: int = 0

        self.cluster_attempts = np.zeros((self.n_clusters, 3), dtype=np.int32)  # for L1..L3
        self.cluster_successes = np.zeros((self.n_clusters, 3), dtype=np.int32)

        # Current target attempt tracking (for L1..L3)
        self.target_tried_mask = np.zeros(3, dtype=np.int32)
        self.target_attempts = 0

        # Logging info for analysis
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

        # Normalize attempts per level within this cluster
        attempts = self.cluster_attempts[cid].astype(np.float32)
        denom = max(1.0, float(attempts.sum()))
        attempts_norm = attempts / denom

        successes = self.cluster_successes[cid].astype(np.float32)
        success_rate = successes / (attempts + 1e-6)

        global_success_rate = np.array([
            self.total_solved / max(1, self.cur_target)
        ], dtype=np.float32)

        obs = np.concatenate([
            progress,
            cid_oh,
            spent_ratio,
            remain_ratio,
            tried_mask,
            attempts_norm,
            success_rate,
            global_success_rate,
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
            "cluster_types": self.cluster_types,  # NOTE: for debugging only
        }
        return obs, info

    def _advance_target(self):
        self.cur_target += 1
        self.target_tried_mask[:] = 0
        self.target_attempts = 0

    def step(self, action: int):
        assert self.action_space.contains(action)

        # Terminal if already finished
        if self.cur_target >= self.n_targets:
            return self._get_obs(), 0.0, True, False, {"reason": "done"}

        cost = self.cfg.costs[int(action)]
        remaining = self.cfg.budget - self.spent

        # Hard budget constraint
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

        # Spend budget
        self.spent += cost
        self._episode_cost += cost

        cid = self._cluster_id(self.cur_target)
        hidden_type = self.cluster_types[cid]

        reward = 0.0
        solved_now = False
        invalid_repeat = False

        if action == 0:
            # Skip target (no success)
            self._advance_target()
        else:
            level = int(action)  # 1..3

            # Track repeats on same target
            mask_idx = level - 1
            if self.target_tried_mask[mask_idx] == 1:
                invalid_repeat = True
            self.target_tried_mask[mask_idx] = 1
            self.target_attempts += 1

            # Update per-cluster stats
            self.cluster_attempts[cid, mask_idx] += 1

            # Sample success
            p = self.cfg.success_probs[hidden_type][level]
            if float(self._np_rng.random()) < p:
                solved_now = True
                self.total_solved += 1
                self.cluster_successes[cid, mask_idx] += 1
                self._advance_target()
            else:
                # Failure: stay on same target unless attempts exceed limit
                if self.target_attempts >= self.cfg.max_attempts_per_target:
                    self._advance_target()

        # Reward shaping
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
            # For analysis (not visible to agent)
            "cluster_id": cid,
            "cluster_type": hidden_type,
        }

        return self._get_obs(), float(reward), terminated, False, info


# -----------------------------
# Baseline policies
# -----------------------------


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
    """Within the same target: try L1 -> L2 -> L3; after that SKIP.

    We infer whether we are still on same target by reading the tried_mask from obs.
    Obs layout includes tried_mask right after (progress + onehot + spent + remain).

    tried_mask = [tried_L1, tried_L2, tried_L3]
    """

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def act(self, obs: np.ndarray) -> int:
        # Parse tried_mask
        idx = 1 + self.n_clusters + 1 + 1
        tried = obs[idx : idx + 3]
        tried = (tried > 0.5).astype(np.int32)

        if tried[0] == 0:
            return 1
        if tried[1] == 0:
            return 2
        if tried[2] == 0:
            return 3
        return 0  # already tried all -> skip


class BudgetAwareGreedy(BaselinePolicy):
    """Simple rule using remaining budget ratio + per-target fallback.

    - If remaining budget is very low: prefer SKIP unless already invested.
    - If remaining budget is moderate: start from L1, but jump to L2 if cluster seems hard.
    - If remaining budget is high: behave like greedy fallback.

    Cluster hardness proxy: if cluster L1 success_rate is low AND enough attempts, jump to L2.
    """

    def __init__(self, n_clusters: int, min_attempts_for_infer: int = 3):
        self.n_clusters = n_clusters
        self.min_attempts_for_infer = min_attempts_for_infer

    def act(self, obs: np.ndarray) -> int:
        # Indices
        progress_end = 1
        onehot_end = progress_end + self.n_clusters
        spent_idx = onehot_end
        remain_idx = onehot_end + 1
        tried_idx = remain_idx + 1
        attempts_norm_idx = tried_idx + 3
        success_rate_idx = attempts_norm_idx + 3

        remain_ratio = float(obs[remain_idx])

        tried = (obs[tried_idx : tried_idx + 3] > 0.5).astype(np.int32)

        # If already trying current target, just fallback
        if tried.sum() > 0:
            if tried[0] == 0:
                return 1
            if tried[1] == 0:
                return 2
            if tried[2] == 0:
                return 3
            return 0

        # Read cluster attempts_norm and success_rate
        # attempts_norm doesn't tell absolute attempts; we approximate via distribution only.
        # So we use success_rate heuristics cautiously.
        l1_succ = float(obs[success_rate_idx + 0])

        if remain_ratio < 0.15:
            # Budget is tight: skip new targets often
            return 0

        if remain_ratio < 0.35:
            # Mid budget: if cluster looks hard, jump to L2
            if l1_succ < 0.2:
                return 2
            return 1

        # Plenty budget: default L1 start
        return 1


# -----------------------------
# Utilities: rollouts & evaluation
# -----------------------------


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


# -----------------------------
# RL: stable-baselines3
# -----------------------------


def make_sb3_env(cfg: EnvConfig):
    # SB3 expects a callable that returns an env instance.
    def _thunk():
        return AnalysisBudgetEnv(cfg)

    return _thunk


def train_rl(algo: str, cfg: EnvConfig, timesteps: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    try:
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
    except Exception as e:
        raise RuntimeError(
            "Please install stable-baselines3: pip install stable-baselines3"
        ) from e

    # Wrap env for logging
    def _make():
        env = AnalysisBudgetEnv(cfg)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([_make])
    eval_env = DummyVecEnv([_make])

    # Simple callback to store learning curve
    class RewardLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ep_returns = []

        def _on_step(self) -> bool:
            # Monitor writes episode info into infos
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self.ep_returns.append(info["episode"]["r"])
            return True


    curve_cb = RewardLogger()

    # if the eval mean reward does not improve for 'patience' evals, stop training
    early_stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=8,   # allow up to 8 evals without improvement
        min_evals=6,                  # allow at least 6 evals before stopping
        verbose=1
    )

    eval_cb = EvalCallback(
        eval_env,
        callback_after_eval=early_stop_cb,
        eval_freq=10_000,         
        n_eval_episodes=30,            
        deterministic=True,
        best_model_save_path=out_dir,
        log_path=out_dir,
        verbose=1
    )

    cb = CallbackList([curve_cb, eval_cb])


    if algo.lower() == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            seed=cfg.seed,
            device="cpu"
        )
    elif algo.lower() == "dqn":
        model = DQN(
            "MlpPolicy",
            vec_env,
            verbose=1,
            buffer_size=100_000,
            learning_rate=2e-4,
            learning_starts=5_000,
            batch_size=256,
            gamma=0.99,
            train_freq=4,
            target_update_interval=2_000,
            seed=cfg.seed,
            device="cpu"
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    model.learn(total_timesteps=timesteps, callback=cb)

    model_path = os.path.join(out_dir, "model.zip")
    model.save(model_path)

    # Save learning curve
    np.save(os.path.join(out_dir, "learning_curve.npy"), np.array(curve_cb.ep_returns, dtype=np.float32))

    return model_path


def load_rl(algo: str, model_path: str):
    try:
        from stable_baselines3 import PPO, DQN
    except Exception as e:
        raise RuntimeError(
            "Please install stable-baselines3: pip install stable-baselines3"
        ) from e

    if algo.lower() == "ppo":
        return PPO.load(model_path)
    if algo.lower() == "dqn":
        return DQN.load(model_path)
    raise ValueError(f"Unknown algo: {algo}")


class SB3PolicyAdapter(BaselinePolicy):
    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def act(self, obs: np.ndarray) -> int:
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a)


# -----------------------------
# Plotting
# -----------------------------


def plot_learning_curve(curve_path: str, out_png: str):
    import matplotlib.pyplot as plt

    y = np.load(curve_path)
    if y.size == 0:
        print("[warn] empty learning curve")
        return

    # Moving average for readability
    win = max(1, min(200, y.size // 20))
    ma = np.convolve(y, np.ones(win) / win, mode="valid")

    plt.figure()
    plt.plot(ma)
    plt.xlabel("Episode (smoothed)")
    plt.ylabel("Episode return")
    plt.title("Learning curve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_solved_vs_budget(results: Dict[str, Dict[int, Dict[str, float]]], out_png: str):
    import matplotlib.pyplot as plt

    budgets = sorted(next(iter(results.values())).keys())

    plt.figure()
    for name, by_budget in results.items():
        means = [by_budget[b]["solved_mean"] for b in budgets]
        stds = [by_budget[b]["solved_std"] for b in budgets]
        plt.errorbar(budgets, means, yerr=stds, capsize=3, label=name)

    plt.xlabel("Budget")
    plt.ylabel("Solved targets (mean ± std)")
    plt.title("Solved vs budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_policy_behavior(env_cfg: EnvConfig, policy: BaselinePolicy, out_png: str, n_episodes: int = 300, seed0: int = 0):
    """Rough interpretability: how often policy uses expensive actions at different remaining budget bins."""
    import matplotlib.pyplot as plt

    env = AnalysisBudgetEnv(env_cfg)

    # bins of remaining ratio
    bins = np.linspace(0.0, 1.0, 6)
    counts = np.zeros(len(bins) - 1, dtype=np.int32)
    expensive = np.zeros(len(bins) - 1, dtype=np.int32)

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + i)
        policy.reset()
        done = False
        while not done:
            # parse remain_ratio
            remain_ratio_idx = 1 + env_cfg.n_clusters + 1
            rr = float(obs[remain_ratio_idx])
            b = int(np.digitize(rr, bins) - 1)
            b = max(0, min(b, len(counts) - 1))

            a = policy.act(obs)
            if a in (2, 3):
                expensive[b] += 1
            counts[b] += 1

            obs, _, term, trunc, _ = env.step(a)
            done = bool(term or trunc)

    frac = np.divide(expensive, np.maximum(1, counts))
    centers = (bins[:-1] + bins[1:]) / 2

    plt.figure()
    plt.plot(centers, frac, marker="o")
    plt.xlabel("Remaining budget ratio (binned)")
    plt.ylabel("P(action in {L2,L3})")
    plt.title("Policy behavior vs remaining budget")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# Main
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity", "train", "sweep"], required=True)
    ap.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--out", type=str, default="out")

    ap.add_argument("--n_clusters", type=int, default=4)
    ap.add_argument("--tpc", type=int, default=5, help="targets per cluster")
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    cfg = EnvConfig(
        n_clusters=args.n_clusters,
        targets_per_cluster=args.tpc,
        budget=args.budget,
        seed=args.seed,
    )

    if args.mode == "sanity":
        env = AnalysisBudgetEnv(cfg)
        policies = {
            "always_l1": AlwaysL1(),
            "always_l3": AlwaysL3(),
            "random": RandomPolicy(seed=cfg.seed),
            "greedy_fallback": GreedyFallback(n_clusters=cfg.n_clusters),
            "budget_aware": BudgetAwareGreedy(n_clusters=cfg.n_clusters),
        }
        for name, pol in policies.items():
            m = eval_policy(cfg, pol, n_episodes=200, seed0=123)
            print(
            f"{name:15s} "
            f"solved_mean={m['solved_mean']:.2f}±{m['solved_std']:.2f} "
            f"spent_mean={m['spent_mean']:.1f}±{m['spent_std']:.1f} "
            f"return_mean={m['return_mean']:.3f}±{m['return_std']:.3f}"
            )

        return

    if args.mode == "train":
        model_path = train_rl(args.algo, cfg, args.timesteps, args.out)
        print(f"Saved model to: {model_path}")

        curve = os.path.join(args.out, "learning_curve.npy")
        if os.path.exists(curve):
            plot_learning_curve(curve, os.path.join(args.out, "learning_curve.png"))
            print(f"Saved learning curve to: {os.path.join(args.out, 'learning_curve.png')}")

        # Behavior plot
        try:
            model = load_rl(args.algo, model_path)
            pol = SB3PolicyAdapter(model)
            plot_policy_behavior(cfg, pol, os.path.join(args.out, "policy_behavior.png"))
            print(f"Saved behavior plot to: {os.path.join(args.out, 'policy_behavior.png')}")
        except Exception as e:
            print(f"[warn] could not plot behavior: {e}")

        return

    if args.mode == "sweep":
        if not args.model:
            raise ValueError("--model is required in sweep mode")

        model = load_rl(args.algo, args.model)
        rl_policy = SB3PolicyAdapter(model)

        policies: Dict[str, BaselinePolicy] = {
            "Fixed-L1": AlwaysL1(),
            "Fixed-L3": AlwaysL3(),
            "Greedy-fallback": GreedyFallback(n_clusters=cfg.n_clusters),
            "Budget-aware": BudgetAwareGreedy(n_clusters=cfg.n_clusters),
            f"RL-{args.algo.upper()}": rl_policy,
        }

        budgets = [20, 40, 60, 80, 100]
        results: Dict[str, Dict[int, Dict[str, float]]] = {name: {} for name in policies}

        for B in budgets:
            cfgB = dataclasses.replace(cfg, budget=B)
            print(f"\n=== Budget = {B} ===")
            for name, pol in policies.items():
                m = eval_policy(cfgB, pol, n_episodes=300, seed0=999)
                results[name][B] = m
                print(
                f"{name:15s} "
                f"solved_mean={m['solved_mean']:.2f}±{m['solved_std']:.2f} "
                f"spent_mean={m['spent_mean']:.1f}±{m['spent_std']:.1f} "
                f"return_mean={m['return_mean']:.3f}±{m['return_std']:.3f}"
            )


        os.makedirs(args.out, exist_ok=True)
        out_png = os.path.join(args.out, "solved_vs_budget.png")
        plot_solved_vs_budget(results, out_png)
        print(f"\nSaved: {out_png}")

        # Behavior plot for RL
        plot_policy_behavior(dataclasses.replace(cfg, budget=max(budgets)), rl_policy, os.path.join(args.out, "policy_behavior_rl.png"))
        print(f"Saved: {os.path.join(args.out, 'policy_behavior_rl.png')}")


if __name__ == "__main__":
    main()
