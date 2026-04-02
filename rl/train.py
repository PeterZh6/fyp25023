"""RL training and evaluation entry point (PPO via stable-baselines3).

Supports:
  - Single binary training with multiple seeds
  - Lambda sweep for Pareto front
  - Info ablation experiments
  - Cross-binary generalization
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from rl.budget_env import AnalysisBudgetEnv, EnvConfig
from rl.baselines import (
    ALL_BASELINES, PolicyFn, evaluate_policy, run_all_baselines,
)


@dataclass
class TrainConfig:
    env_config_path: str = "rl/configs/env_configs.yaml"
    binary_name: str = "gcc"
    budget_ratio: float = 2.0

    algorithm: str = "PPO"
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    policy_net: str = "MlpPolicy"
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [64, 64]})

    num_seeds: int = 5
    cost_lambda: float = 0.02
    eval_freq: int = 10_000
    eval_episodes: int = 100
    save_dir: str = "results/"

    use_site_features: bool = True
    use_global_stats: bool = True
    oracle_mode: bool = False


def make_env_config(train_cfg: TrainConfig) -> EnvConfig:
    env_config = EnvConfig.from_yaml(train_cfg.env_config_path, train_cfg.binary_name)
    env_config.budget_ratio = train_cfg.budget_ratio
    env_config.budget = env_config.budget_ratio * env_config.num_sites * env_config.costs["L1"]
    env_config.cost_lambda = train_cfg.cost_lambda
    env_config.oracle_mode = train_cfg.oracle_mode
    env_config.use_site_features = train_cfg.use_site_features
    env_config.use_global_stats = train_cfg.use_global_stats
    return env_config


class SB3PolicyAdapter:
    """Wraps an SB3 model to match the PolicyFn signature."""

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def __call__(self, obs: np.ndarray, env: AnalysisBudgetEnv) -> int:
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a)


def train_single(train_cfg: TrainConfig, seed: int) -> Dict[str, Any]:
    """Train a single model with given config and seed.

    Returns dict with model_path, best_mean_reward, and learning_curve.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

    env_config = make_env_config(train_cfg)
    run_dir = os.path.join(train_cfg.save_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    def _make_env():
        return Monitor(AnalysisBudgetEnv(env_config))

    vec_env = DummyVecEnv([_make_env])
    eval_env = DummyVecEnv([_make_env])

    class RewardLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ep_returns: list[float] = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self.ep_returns.append(info["episode"]["r"])
            return True

    curve_cb = RewardLogger()

    eval_cb = EvalCallback(
        eval_env,
        eval_freq=train_cfg.eval_freq,
        n_eval_episodes=train_cfg.eval_episodes,
        deterministic=True,
        best_model_save_path=run_dir,
        log_path=run_dir,
        verbose=0,
    )

    model = PPO(
        train_cfg.policy_net, vec_env,
        seed=seed,
        learning_rate=train_cfg.learning_rate,
        n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size,
        n_epochs=train_cfg.n_epochs,
        gamma=train_cfg.gamma,
        policy_kwargs=train_cfg.policy_kwargs,
        verbose=0,
        device="cpu",
    )

    from stable_baselines3.common.callbacks import CallbackList
    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        callback=CallbackList([curve_cb, eval_cb]),
    )

    model_path = os.path.join(run_dir, "final_model.zip")
    model.save(model_path)

    curve = np.array(curve_cb.ep_returns, dtype=np.float32)
    np.save(os.path.join(run_dir, "learning_curve.npy"), curve)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(train_cfg), f, indent=2, default=str)

    return {
        "model_path": model_path,
        "best_model_path": os.path.join(run_dir, "best_model.zip"),
        "learning_curve": curve,
        "seed": seed,
    }


def evaluate_trained_model(
    model_path: str,
    train_cfg: TrainConfig,
    n_episodes: int = 1000,
    seed: int = 42,
    test_binary: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model against all baselines."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    rl_policy = SB3PolicyAdapter(model)

    if test_binary:
        cfg = dataclasses.replace(train_cfg, binary_name=test_binary)
    else:
        cfg = train_cfg
    env_config = make_env_config(cfg)
    env = AnalysisBudgetEnv(env_config)

    rl_metrics = evaluate_policy(rl_policy, env, n_episodes=n_episodes, seed=seed)

    baseline_metrics = {}
    for name, policy_fn in ALL_BASELINES.items():
        baseline_metrics[name] = evaluate_policy(
            policy_fn, env, n_episodes=n_episodes, seed=seed
        )

    best_baseline_name = max(
        baseline_metrics, key=lambda k: baseline_metrics[k]["mean_resolve_rate"]
    )
    best_bl = baseline_metrics[best_baseline_name]

    improvement = rl_metrics["mean_resolve_rate"] - best_bl["mean_resolve_rate"]

    return {
        "rl": rl_metrics,
        "baselines": baseline_metrics,
        "best_baseline": best_baseline_name,
        "improvement_over_best": improvement,
        "binary": cfg.binary_name,
    }


# ---- Experiment suites ----

def run_experiment_a(train_cfg: TrainConfig, binaries: List[str]) -> Dict[str, Any]:
    """Experiment A: Single binary training + multi seed."""
    results: Dict[str, Any] = {}
    for binary in binaries:
        print(f"\n{'='*60}")
        print(f"Experiment A: {binary}")
        print(f"{'='*60}")
        cfg = dataclasses.replace(
            train_cfg,
            binary_name=binary,
            save_dir=os.path.join(train_cfg.save_dir, "exp_a", binary),
        )
        seed_results = []
        for s in range(cfg.num_seeds):
            print(f"  Training seed {s}...")
            result = train_single(cfg, seed=s)
            eval_result = evaluate_trained_model(
                result["best_model_path"], cfg, n_episodes=200
            )
            seed_results.append(eval_result)
            print(f"    RL resolve_rate={eval_result['rl']['mean_resolve_rate']:.3f}, "
                  f"best_baseline={eval_result['best_baseline']} "
                  f"({eval_result['baselines'][eval_result['best_baseline']]['mean_resolve_rate']:.3f})")

        results[binary] = seed_results
    return results


def run_experiment_b(train_cfg: TrainConfig) -> Dict[str, Any]:
    """Experiment B: Lambda sweep for Pareto front."""
    lambda_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results: Dict[float, List[Dict]] = {}

    for lam in lambda_values:
        print(f"\n--- Lambda = {lam} ---")
        cfg = dataclasses.replace(
            train_cfg,
            cost_lambda=lam,
            num_seeds=3,
            save_dir=os.path.join(train_cfg.save_dir, "exp_b", f"lambda_{lam}"),
        )
        seed_results = []
        for s in range(cfg.num_seeds):
            result = train_single(cfg, seed=s)
            eval_result = evaluate_trained_model(
                result["best_model_path"], cfg, n_episodes=200
            )
            seed_results.append(eval_result)
        results[lam] = seed_results

        mean_rate = np.mean([r["rl"]["mean_resolve_rate"] for r in seed_results])
        mean_cost = np.mean([r["rl"]["mean_cost"] for r in seed_results])
        print(f"  resolve_rate={mean_rate:.3f}, cost={mean_cost:.1f}")

    return results


def run_experiment_c(train_cfg: TrainConfig) -> Dict[str, Any]:
    """Experiment C: Info ablation."""
    configs = [
        {"name": "full", "use_site_features": True, "use_global_stats": True, "oracle_mode": False},
        {"name": "no_site_features", "use_site_features": False, "use_global_stats": True, "oracle_mode": False},
        {"name": "minimal", "use_site_features": False, "use_global_stats": False, "oracle_mode": False},
        {"name": "oracle", "use_site_features": True, "use_global_stats": True, "oracle_mode": True},
    ]
    results: Dict[str, Any] = {}

    for ablation in configs:
        name = ablation.pop("name")
        print(f"\n--- Ablation: {name} ---")
        cfg = dataclasses.replace(
            train_cfg,
            save_dir=os.path.join(train_cfg.save_dir, "exp_c", name),
            **ablation,
        )
        seed_results = []
        for s in range(cfg.num_seeds):
            result = train_single(cfg, seed=s)
            eval_result = evaluate_trained_model(
                result["best_model_path"], cfg, n_episodes=200
            )
            seed_results.append(eval_result)
        results[name] = seed_results

        mean_rate = np.mean([r["rl"]["mean_resolve_rate"] for r in seed_results])
        print(f"  resolve_rate={mean_rate:.3f}")

    return results


def run_experiment_d(
    train_cfg: TrainConfig,
    all_binaries: List[str],
) -> Dict[str, Any]:
    """Experiment D: Cross-binary generalization."""
    splits = {
        "A": {"train": "gcc", "test": ["ssh", "ssh-keygen", "h264ref"]},
        "B": {"train": "h264ref", "test": ["gcc", "openssl", "ssh"]},
        "C": {"train": "mixed_no_cpp", "test": ["gcc", "ssh", "openssl", "h264ref", "bzip2"]},
    }
    results: Dict[str, Any] = {}

    for split_name, split in splits.items():
        print(f"\n{'='*60}")
        print(f"Experiment D, split {split_name}: train={split['train']}")
        print(f"{'='*60}")
        cfg = dataclasses.replace(
            train_cfg,
            binary_name=split["train"],
            num_seeds=3,
            save_dir=os.path.join(train_cfg.save_dir, "exp_d", split_name),
        )

        trained_models = []
        for s in range(cfg.num_seeds):
            result = train_single(cfg, seed=s)
            trained_models.append(result["best_model_path"])

        split_results: Dict[str, Any] = {"train": split["train"], "test_results": {}}
        for test_binary in split["test"]:
            seed_evals = []
            for model_path in trained_models:
                eval_result = evaluate_trained_model(
                    model_path, cfg, n_episodes=200, test_binary=test_binary
                )
                seed_evals.append(eval_result)
            split_results["test_results"][test_binary] = seed_evals
            mean_rate = np.mean([r["rl"]["mean_resolve_rate"] for r in seed_evals])
            best_bl = seed_evals[0]["best_baseline"]
            bl_rate = np.mean([
                r["baselines"][best_bl]["mean_resolve_rate"] for r in seed_evals
            ])
            print(f"  {test_binary}: RL={mean_rate:.3f}, best_baseline={best_bl}({bl_rate:.3f})")

        results[split_name] = split_results

    return results


# ---- CLI ----

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL training for budget allocation")
    parser.add_argument("--mode", choices=["train", "eval", "suite_a", "suite_b", "suite_c", "suite_d", "sanity"],
                        required=True)

    parser.add_argument("--config", default="rl/configs/env_configs.yaml")
    parser.add_argument("--binary", default="gcc")
    parser.add_argument("--budget-ratio", type=float, default=2.0)

    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cost-lambda", type=float, default=0.02)
    parser.add_argument("--num-seeds", type=int, default=5)

    parser.add_argument("--no-site-features", action="store_true")
    parser.add_argument("--no-global-stats", action="store_true")
    parser.add_argument("--oracle", action="store_true")

    parser.add_argument("--eval-model", type=str, default=None)
    parser.add_argument("--save-dir", default="results/")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    train_cfg = TrainConfig(
        env_config_path=args.config,
        binary_name=args.binary,
        budget_ratio=args.budget_ratio,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        cost_lambda=args.cost_lambda,
        num_seeds=args.num_seeds,
        use_site_features=not args.no_site_features,
        use_global_stats=not args.no_global_stats,
        oracle_mode=args.oracle,
        save_dir=args.save_dir,
    )

    if args.mode == "sanity":
        print("Running sanity check: short training + baseline comparison")
        cfg = dataclasses.replace(train_cfg, total_timesteps=50_000, num_seeds=1,
                                  save_dir=os.path.join(args.save_dir, "sanity"))
        result = train_single(cfg, seed=0)
        eval_result = evaluate_trained_model(result["best_model_path"], cfg, n_episodes=200)
        print(f"\nRL resolve_rate: {eval_result['rl']['mean_resolve_rate']:.3f}")
        print(f"Best baseline: {eval_result['best_baseline']} "
              f"({eval_result['baselines'][eval_result['best_baseline']]['mean_resolve_rate']:.3f})")
        print(f"Improvement: {eval_result['improvement_over_best']:+.3f}")

    elif args.mode == "train":
        for s in range(train_cfg.num_seeds):
            print(f"\n--- Seed {s} ---")
            result = train_single(train_cfg, seed=s)
            eval_result = evaluate_trained_model(result["best_model_path"], train_cfg)
            print(f"  resolve_rate={eval_result['rl']['mean_resolve_rate']:.3f}")

            out_path = os.path.join(train_cfg.save_dir, f"seed_{s}", "eval_results.json")
            with open(out_path, "w") as f:
                json.dump(eval_result, f, indent=2)

    elif args.mode == "eval":
        if not args.eval_model:
            raise ValueError("--eval-model required in eval mode")
        eval_result = evaluate_trained_model(args.eval_model, train_cfg)
        print(json.dumps(eval_result, indent=2))

    elif args.mode == "suite_a":
        binaries = ["gcc", "ssh", "openssl", "h264ref", "bzip2"]
        results = run_experiment_a(train_cfg, binaries)
        with open(os.path.join(args.save_dir, "exp_a_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    elif args.mode == "suite_b":
        results = run_experiment_b(train_cfg)
        with open(os.path.join(args.save_dir, "exp_b_results.json"), "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)

    elif args.mode == "suite_c":
        results = run_experiment_c(train_cfg)
        with open(os.path.join(args.save_dir, "exp_c_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    elif args.mode == "suite_d":
        all_binaries = ["gcc", "ssh", "openssl", "h264ref", "bzip2", "ssh-keygen"]
        results = run_experiment_d(train_cfg, all_binaries)
        with open(os.path.join(args.save_dir, "exp_d_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
