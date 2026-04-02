"""Plotting utilities for RL experiment analysis."""

from __future__ import annotations

from typing import Dict

import numpy as np


def plot_learning_curve(curve_path: str, out_png: str):
    import matplotlib.pyplot as plt

    y = np.load(curve_path)
    if y.size == 0:
        print("[warn] empty learning curve")
        return

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
    print(f"Saved: {out_png}")


def plot_solved_vs_budget(results: Dict[str, Dict[int, Dict[str, float]]], out_png: str):
    import matplotlib.pyplot as plt

    budgets = sorted(next(iter(results.values())).keys())

    plt.figure()
    for name, by_budget in results.items():
        means = [by_budget[b]["solved_mean"] for b in budgets]
        stds = [by_budget[b]["solved_std"] for b in budgets]
        plt.errorbar(budgets, means, yerr=stds, capsize=3, label=name)

    plt.xlabel("Budget")
    plt.ylabel("Solved targets (mean \u00b1 std)")
    plt.title("Solved vs budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"Saved: {out_png}")


def plot_policy_behavior(env_cfg, policy, out_png: str, n_episodes: int = 300, seed0: int = 0):
    """P(use expensive action) vs remaining budget ratio."""
    import matplotlib.pyplot as plt
    from rl.budget_env import AnalysisBudgetEnv

    env = AnalysisBudgetEnv(env_cfg)

    bins = np.linspace(0.0, 1.0, 6)
    counts = np.zeros(len(bins) - 1, dtype=np.int32)
    expensive = np.zeros(len(bins) - 1, dtype=np.int32)

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + i)
        policy.reset()
        done = False
        while not done:
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
    print(f"Saved: {out_png}")
