"""Plotting utilities for RL experiment analysis.

Generates:
  1. Difficulty Distribution Overview (stacked bar)
  2. Success Rate Heatmap
  3. Pareto Front (cost vs resolve rate)
  4. RL vs Baselines Per Binary (grouped bar)
  5. Cross-Binary Generalization Heatmap
  6. Info Ablation (bar chart)
  7. Learning Curves
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

FIGSIZE = (8, 5)
FONT_SIZE = 12
COLORS = {
    "easy": "#4CAF50",
    "medium": "#FF9800",
    "hard": "#F44336",
    "rl": "#2196F3",
    "baseline": "#9E9E9E",
    "oracle": "#9C27B0",
}
BASELINE_MARKERS = {
    "all_skip": "x", "all_l1": "s", "all_l2": "D",
    "all_l3": "^", "random": "o", "greedy_cheap": "v",
    "budget_aware": "P", "escalation": "*",
}
BASELINE_COLORS = {
    "all_skip": "#795548", "all_l1": "#FF5722", "all_l2": "#FF9800",
    "all_l3": "#FFC107", "random": "#9E9E9E", "greedy_cheap": "#8BC34A",
    "budget_aware": "#00BCD4", "escalation": "#E91E63",
}


def _save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    pdf_path = path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"Saved: {path} + {pdf_path}")


def plot_difficulty_distributions(dist_path: str, out_path: str = "results/figures/difficulty_dist.png"):
    """Fig 1: Stacked bar chart of difficulty distributions."""
    import matplotlib.pyplot as plt

    with open(dist_path) as f:
        data = json.load(f)

    per_binary = data["per_binary"]
    items = sorted(per_binary.items(), key=lambda x: x[1]["hard"], reverse=True)

    names = [n.split("_base.")[0] if "_base." in n else n for n, _ in items]
    easy = [v["easy"] for _, v in items]
    medium = [v["medium"] for _, v in items]
    hard = [v["hard"] for _, v in items]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.4), 5))
    x = np.arange(len(names))
    w = 0.7

    ax.bar(x, easy, w, label="Easy", color=COLORS["easy"])
    ax.bar(x, medium, w, bottom=easy, label="Medium", color=COLORS["medium"])
    ax.bar(x, hard, w, bottom=np.array(easy) + np.array(medium), label="Hard", color=COLORS["hard"])

    cpp_indices = [i for i, (n, _) in enumerate(items) if "xalan" in n.lower() or "dealii" in n.lower()]
    for idx in cpp_indices:
        ax.get_children()[idx].set_edgecolor("black")
        ax.get_children()[idx].set_linewidth(2)
        ax.annotate("C++", (idx, 1.02), ha="center", fontsize=8, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Proportion")
    ax.set_title("Difficulty Distribution per Binary")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)

    _save_fig(fig, out_path)


def plot_success_rate_heatmap(rates_path: str, out_path: str = "results/figures/success_heatmap.png"):
    """Fig 2: Success rate matrix heatmap."""
    import matplotlib.pyplot as plt

    with open(rates_path) as f:
        data = json.load(f)

    agg = data.get("aggregate_no_cpp", data.get("aggregate_all", {}))
    levels = ["SKIP", "L1", "L2", "L3"]
    diffs = ["easy", "medium", "hard"]

    matrix = np.zeros((4, 3))
    mask = np.zeros((4, 3), dtype=bool)

    for i, lvl in enumerate(levels):
        if lvl == "SKIP":
            matrix[i, :] = 0
            continue
        rates = agg.get(lvl, {})
        for j, d in enumerate(diffs):
            val = rates.get(d)
            if val is None:
                matrix[i, j] = 0.5
                mask[i, j] = True
            else:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    for i in range(4):
        for j in range(3):
            val = matrix[i, j]
            text = f"{val:.2f}" if not mask[i, j] else f"{val:.2f}*"
            color = "white" if val < 0.3 or val > 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)
            if mask[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=2, edgecolor="blue",
                                     facecolor="none", linestyle="--")
                ax.add_patch(rect)

    ax.set_xticks(range(3))
    ax.set_xticklabels(diffs)
    ax.set_yticks(range(4))
    ax.set_yticklabels(levels)
    ax.set_title("Success Rate Matrix (* = estimated)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    _save_fig(fig, out_path)


def plot_pareto_front(
    lambda_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    out_path: str = "results/figures/pareto_front.png",
    title: str = "Pareto Front: Cost vs Resolve Rate",
):
    """Fig 3: Pareto front with RL lambda sweep + baseline points."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for bl_name, metrics in baseline_results.items():
        if bl_name in ("n_episodes", "num_sites", "budget", "budget_ratio"):
            continue
        marker = BASELINE_MARKERS.get(bl_name, "o")
        color = BASELINE_COLORS.get(bl_name, "#9E9E9E")
        ax.scatter(
            metrics["mean_cost"], metrics["mean_resolve_rate"],
            marker=marker, color=color, s=100, zorder=5,
            label=bl_name, edgecolors="black", linewidths=0.5,
        )

    if lambda_results:
        lam_vals = sorted(lambda_results.keys(), key=float)
        costs_mean, rates_mean, rates_std = [], [], []
        for lam in lam_vals:
            seed_results = lambda_results[lam]
            c = np.mean([r["rl"]["mean_cost"] for r in seed_results])
            r_mean = np.mean([r["rl"]["mean_resolve_rate"] for r in seed_results])
            r_std = np.std([r["rl"]["mean_resolve_rate"] for r in seed_results])
            costs_mean.append(c)
            rates_mean.append(r_mean)
            rates_std.append(r_std)

        ax.errorbar(
            costs_mean, rates_mean, yerr=rates_std,
            fmt="-o", color=COLORS["rl"], capsize=3, linewidth=2,
            markersize=8, label="RL (λ sweep)", zorder=10,
        )
        for i, lam in enumerate(lam_vals):
            ax.annotate(f"λ={lam}", (costs_mean[i], rates_mean[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Mean Cost per Episode")
    ax.set_ylabel("Mean Resolve Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, out_path)


def plot_rl_vs_baselines(
    eval_results: Dict[str, List[Dict]],
    out_path: str = "results/figures/rl_vs_baselines.png",
):
    """Fig 4: Grouped bar chart of RL vs baselines per binary."""
    import matplotlib.pyplot as plt

    binaries = list(eval_results.keys())
    n = len(binaries)

    fig, ax = plt.subplots(figsize=(max(8, n * 2), 5))

    x = np.arange(n)
    width = 0.25

    rl_means, rl_stds = [], []
    bl_means, bl_stds = [], []
    bl_names_list = []

    for binary in binaries:
        seed_results = eval_results[binary]
        rl_rates = [r["rl"]["mean_resolve_rate"] for r in seed_results]
        rl_means.append(np.mean(rl_rates))
        rl_stds.append(np.std(rl_rates))

        best_bl = seed_results[0]["best_baseline"]
        bl_names_list.append(best_bl)
        bl_rates = [r["baselines"][best_bl]["mean_resolve_rate"] for r in seed_results]
        bl_means.append(np.mean(bl_rates))
        bl_stds.append(np.std(bl_rates))

    ax.bar(x - width / 2, bl_means, width, yerr=bl_stds, capsize=3,
           label="Best Baseline", color=COLORS["baseline"], edgecolor="black")
    ax.bar(x + width / 2, rl_means, width, yerr=rl_stds, capsize=3,
           label="RL (PPO)", color=COLORS["rl"], edgecolor="black")

    short_names = [b.split("_base.")[0] if "_base." in b else b for b in binaries]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("Resolve Rate")
    ax.set_title("RL vs Best Baseline per Binary")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    _save_fig(fig, out_path)


def plot_cross_binary_heatmap(
    cross_results: Dict[str, Any],
    out_path: str = "results/figures/cross_binary_heatmap.png",
):
    """Fig 5: Cross-binary generalization heatmap."""
    import matplotlib.pyplot as plt

    splits = list(cross_results.keys())
    all_test_binaries = set()
    for split_data in cross_results.values():
        all_test_binaries.update(split_data["test_results"].keys())
    test_binaries = sorted(all_test_binaries)

    matrix = np.full((len(splits), len(test_binaries)), np.nan)

    for i, split_name in enumerate(splits):
        split_data = cross_results[split_name]
        for j, test_bin in enumerate(test_binaries):
            if test_bin in split_data["test_results"]:
                seed_evals = split_data["test_results"][test_bin]
                rl_rate = np.mean([r["rl"]["mean_resolve_rate"] for r in seed_evals])
                best_bl = seed_evals[0]["best_baseline"]
                bl_rate = np.mean([
                    r["baselines"][best_bl]["mean_resolve_rate"] for r in seed_evals
                ])
                matrix[i, j] = (rl_rate - bl_rate) * 100

    fig, ax = plt.subplots(figsize=(max(6, len(test_binaries) * 1.2), max(4, len(splits) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    for i in range(len(splits)):
        for j in range(len(test_binaries)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}%", ha="center", va="center", fontsize=10)

    short_names = [b.split("_base.")[0] if "_base." in b else b for b in test_binaries]
    train_labels = [f"{s}: {cross_results[s]['train']}" for s in splits]

    ax.set_xticks(range(len(test_binaries)))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels(train_labels)
    ax.set_title("Cross-Binary Generalization\n(RL improvement over best baseline, %)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    _save_fig(fig, out_path)


def plot_info_ablation(
    ablation_results: Dict[str, List[Dict]],
    out_path: str = "results/figures/info_ablation.png",
):
    """Fig 6: Info ablation bar chart."""
    import matplotlib.pyplot as plt

    configs = list(ablation_results.keys())
    means = []
    stds = []
    for cfg_name in configs:
        seed_results = ablation_results[cfg_name]
        rates = [r["rl"]["mean_resolve_rate"] for r in seed_results]
        means.append(np.mean(rates))
        stds.append(np.std(rates))

    colors = ["#2196F3", "#FF9800", "#9E9E9E", "#9C27B0"][:len(configs)]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylabel("Resolve Rate")
    ax.set_title("Information Ablation Study")
    ax.grid(True, axis="y", alpha=0.3)

    _save_fig(fig, out_path)


def plot_learning_curves(
    curve_dir: str,
    out_path: str = "results/figures/learning_curves.png",
    baseline_levels: Optional[Dict[str, float]] = None,
    window: int = 200,
):
    """Fig 7: Learning curves with multi-seed mean ± std."""
    import matplotlib.pyplot as plt

    curves = []
    for fname in sorted(os.listdir(curve_dir)):
        if fname.endswith("learning_curve.npy"):
            curves.append(np.load(os.path.join(curve_dir, fname)))

    if not curves:
        for subdir in sorted(os.listdir(curve_dir)):
            npy = os.path.join(curve_dir, subdir, "learning_curve.npy")
            if os.path.exists(npy):
                curves.append(np.load(npy))

    if not curves:
        print(f"No learning curves found in {curve_dir}")
        return

    min_len = min(len(c) for c in curves)
    truncated = np.array([c[:min_len] for c in curves])

    win = min(window, min_len // 5)
    if win < 1:
        win = 1

    smoothed = np.array([
        np.convolve(row, np.ones(win) / win, mode="valid")
        for row in truncated
    ])

    mean = smoothed.mean(axis=0)
    std = smoothed.std(axis=0)
    x = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, mean, color=COLORS["rl"], linewidth=2, label=f"PPO ({len(curves)} seeds)")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=COLORS["rl"])

    if baseline_levels:
        for name, level in baseline_levels.items():
            color = BASELINE_COLORS.get(name, "#9E9E9E")
            ax.axhline(y=level, color=color, linestyle="--", alpha=0.7, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return")
    ax.set_title("Learning Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    _save_fig(fig, out_path)


def generate_all_figures(results_dir: str = "results/"):
    """Generate all figures from saved experiment results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": FONT_SIZE})

    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    dist_path = "data/calibration/difficulty_distributions.json"
    rates_path = "data/calibration/success_rates.json"

    if os.path.exists(dist_path):
        plot_difficulty_distributions(dist_path, os.path.join(fig_dir, "difficulty_dist.png"))

    if os.path.exists(rates_path):
        plot_success_rate_heatmap(rates_path, os.path.join(fig_dir, "success_heatmap.png"))

    exp_a_path = os.path.join(results_dir, "exp_a_results.json")
    if os.path.exists(exp_a_path):
        with open(exp_a_path) as f:
            exp_a = json.load(f)
        plot_rl_vs_baselines(exp_a, os.path.join(fig_dir, "rl_vs_baselines.png"))

    exp_b_path = os.path.join(results_dir, "exp_b_results.json")
    baseline_path = "data/calibration/baseline_results_gcc.json"
    if os.path.exists(exp_b_path) and os.path.exists(baseline_path):
        with open(exp_b_path) as f:
            exp_b = json.load(f)
        with open(baseline_path) as f:
            bl_data = json.load(f)
        plot_pareto_front(
            exp_b, bl_data.get("baselines", {}),
            os.path.join(fig_dir, "pareto_front.png"),
        )

    exp_c_path = os.path.join(results_dir, "exp_c_results.json")
    if os.path.exists(exp_c_path):
        with open(exp_c_path) as f:
            exp_c = json.load(f)
        plot_info_ablation(exp_c, os.path.join(fig_dir, "info_ablation.png"))

    exp_d_path = os.path.join(results_dir, "exp_d_results.json")
    if os.path.exists(exp_d_path):
        with open(exp_d_path) as f:
            exp_d = json.load(f)
        plot_cross_binary_heatmap(exp_d, os.path.join(fig_dir, "cross_binary_heatmap.png"))

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    generate_all_figures()
