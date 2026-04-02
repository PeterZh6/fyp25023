"""RL training and evaluation entry point (PPO / DQN via stable-baselines3)."""

from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Dict

import numpy as np

from rl.budget_env import AnalysisBudgetEnv, EnvConfig
from rl.baselines import (
    BaselinePolicy, AlwaysL1, AlwaysL3, RandomPolicy,
    GreedyFallback, BudgetAwareGreedy, eval_policy,
)
from rl.plotting import plot_learning_curve, plot_policy_behavior, plot_solved_vs_budget


class SB3PolicyAdapter(BaselinePolicy):
    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def act(self, obs: np.ndarray) -> int:
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a)


def train_rl(algo: str, cfg: EnvConfig, timesteps: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import (
        BaseCallback, EvalCallback,
        StopTrainingOnNoModelImprovement, CallbackList,
    )

    def _make():
        return Monitor(AnalysisBudgetEnv(cfg))

    vec_env = DummyVecEnv([_make])
    eval_env = DummyVecEnv([_make])

    class RewardLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ep_returns = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self.ep_returns.append(info["episode"]["r"])
            return True

    curve_cb = RewardLogger()

    early_stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=8,
        min_evals=6,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        callback_after_eval=early_stop_cb,
        eval_freq=10_000,
        n_eval_episodes=30,
        deterministic=True,
        best_model_save_path=out_dir,
        log_path=out_dir,
        verbose=1,
    )

    cb = CallbackList([curve_cb, eval_cb])

    if algo.lower() == "ppo":
        model = PPO(
            "MlpPolicy", vec_env, verbose=1,
            n_steps=2048, batch_size=256, learning_rate=3e-4,
            gamma=0.99, seed=cfg.seed, device="cpu",
        )
    elif algo.lower() == "dqn":
        model = DQN(
            "MlpPolicy", vec_env, verbose=1,
            buffer_size=100_000, learning_rate=2e-4, learning_starts=5_000,
            batch_size=256, gamma=0.99, train_freq=4,
            target_update_interval=2_000, seed=cfg.seed, device="cpu",
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    model.learn(total_timesteps=timesteps, callback=cb)

    model_path = os.path.join(out_dir, "model.zip")
    model.save(model_path)

    np.save(
        os.path.join(out_dir, "learning_curve.npy"),
        np.array(curve_cb.ep_returns, dtype=np.float32),
    )

    return model_path


def load_rl(algo: str, model_path: str):
    from stable_baselines3 import PPO, DQN

    if algo.lower() == "ppo":
        return PPO.load(model_path)
    if algo.lower() == "dqn":
        return DQN.load(model_path)
    raise ValueError(f"Unknown algo: {algo}")


# ---- CLI ----

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

    ap.add_argument("--cost_lambda", type=float, default=0.02)
    ap.add_argument("--no_cluster_stats", action="store_true",
                    help="Ablation: remove cluster stats from observation")
    ap.add_argument("--oracle", action="store_true",
                    help="Oracle mode: reveal hidden difficulty")

    args = ap.parse_args()

    cfg = EnvConfig(
        n_clusters=args.n_clusters,
        targets_per_cluster=args.tpc,
        budget=args.budget,
        seed=args.seed,
        cost_lambda=args.cost_lambda,
        use_cluster_stats=not args.no_cluster_stats,
        oracle_mode=args.oracle,
    )

    if args.mode == "sanity":
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
                f"solved_mean={m['solved_mean']:.2f}\u00b1{m['solved_std']:.2f} "
                f"spent_mean={m['spent_mean']:.1f}\u00b1{m['spent_std']:.1f} "
                f"return_mean={m['return_mean']:.3f}\u00b1{m['return_std']:.3f}"
            )
        return

    if args.mode == "train":
        model_path = train_rl(args.algo, cfg, args.timesteps, args.out)
        print(f"Saved model to: {model_path}")

        curve = os.path.join(args.out, "learning_curve.npy")
        if os.path.exists(curve):
            plot_learning_curve(curve, os.path.join(args.out, "learning_curve.png"))

        try:
            model = load_rl(args.algo, model_path)
            pol = SB3PolicyAdapter(model)
            plot_policy_behavior(cfg, pol, os.path.join(args.out, "policy_behavior.png"))
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
        results: Dict[str, Dict[int, Dict[str, float]]] = {n: {} for n in policies}

        for B in budgets:
            cfgB = dataclasses.replace(cfg, budget=B)
            print(f"\n=== Budget = {B} ===")
            for name, pol in policies.items():
                m = eval_policy(cfgB, pol, n_episodes=300, seed0=999)
                results[name][B] = m
                print(
                    f"{name:15s} "
                    f"solved_mean={m['solved_mean']:.2f}\u00b1{m['solved_std']:.2f} "
                    f"spent_mean={m['spent_mean']:.1f}\u00b1{m['spent_std']:.1f} "
                    f"return_mean={m['return_mean']:.3f}\u00b1{m['return_std']:.3f}"
                )

        os.makedirs(args.out, exist_ok=True)
        plot_solved_vs_budget(results, os.path.join(args.out, "solved_vs_budget.png"))
        plot_policy_behavior(
            dataclasses.replace(cfg, budget=max(budgets)),
            rl_policy,
            os.path.join(args.out, "policy_behavior_rl.png"),
        )


if __name__ == "__main__":
    main()
