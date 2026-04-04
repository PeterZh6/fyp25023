"""Plots for sensitivity + cross-binary calibration JSON results.

Reads ``data/calibration/sensitivity_results.json`` and
``data/calibration/cross_binary_results.json``, writes PNG+PDF under
``results/figures/`` (same convention as ``rl.plotting._save_fig``).
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Repo root on ``python rl/plot_results.py``
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl.plotting import BASELINE_COLORS, COLORS, FIGSIZE, FONT_SIZE, _save_fig

# gcc / ssh / dealII 3×3 heatmap (short name order)
HEATMAP_SHORT_ORDER = ("gcc", "ssh", "dealII")
HEATMAP_FULL_TO_SHORT: Dict[str, str] = {
    "gcc_base.arm32-gcc81-O3": "gcc",
    "dealII_base.arm32-gcc81-O3": "dealII",
    "ssh": "ssh",
}


def _short_binary_name(full: str) -> str:
    if "_base." in full:
        return full.split("_base.")[0]
    return full


def _max_baseline_resolve_rate(run: Dict[str, Any]) -> float:
    return max(m["mean_resolve_rate"] for m in run["baselines"].values())


def _load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def plot_fig1_baseline_grouped_bars(
    sensitivity_path: str,
    out_path: str = "results/figures/sensitivity_baseline_rl_vs_baselines.png",
):
    """Grouped bars: RL (PPO) vs budget_aware vs all_l1 at cost baseline, per binary."""
    import matplotlib.pyplot as plt

    data = _load_json(sensitivity_path)
    baseline_runs = [r for r in data["runs"] if r.get("cost_name") == "baseline"]
    by_binary: Dict[str, List[Dict]] = defaultdict(list)
    for r in baseline_runs:
        by_binary[r["binary"]].append(r)

    # Stable order: short names alphabetically (5 binaries)
    binaries = sorted(by_binary.keys(), key=_short_binary_name)
    n = len(binaries)
    rl_m, ba_m, l1_m = [], [], []

    for b in binaries:
        runs = by_binary[b]
        rl_m.append(np.mean([x["rl"]["mean_resolve_rate"] for x in runs]))
        ba_m.append(np.mean([x["baselines"]["budget_aware"]["mean_resolve_rate"] for x in runs]))
        l1_m.append(np.mean([x["baselines"]["all_l1"]["mean_resolve_rate"] for x in runs]))

    fig, ax = plt.subplots(figsize=(max(8, n * 1.35), 5))
    x = np.arange(n)
    w = 0.25
    labels = [_short_binary_name(b) for b in binaries]

    bars_rl = ax.bar(
        x - w, rl_m, w, label="RL (PPO)", color=COLORS["rl"], edgecolor="black", linewidth=0.5
    )
    bars_ba = ax.bar(
        x, ba_m, w, label="budget_aware", color=BASELINE_COLORS["budget_aware"], edgecolor="black", linewidth=0.5
    )
    bars_l1 = ax.bar(
        x + w, l1_m, w, label="all_l1", color=BASELINE_COLORS["all_l1"], edgecolor="black", linewidth=0.5
    )

    ymax = max(rl_m + ba_m + l1_m)
    for container, vals in ((bars_rl, rl_m), (bars_ba, ba_m), (bars_l1, l1_m)):
        for rect, v in zip(container, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Resolve rate")
    ax.set_title("Baseline cost: RL vs selected baselines per binary")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, min(1.08, ymax * 1.18))

    _save_fig(fig, out_path)


def plot_fig2_gcc_dealii_cost_sensitivity(
    sensitivity_path: str,
    out_path: str = "results/figures/sensitivity_gcc_dealii_rl_by_cost.png",
):
    """RL resolve rate vs cost preset (baseline/low/high), gcc & dealII, seed std error bars."""
    import matplotlib.pyplot as plt

    data = _load_json(sensitivity_path)
    meta = data.get("meta", {})
    cost_order: List[str] = list(meta.get("costs_run", ["baseline", "low", "high"]))

    targets = {"gcc_base.arm32-gcc81-O3", "dealII_base.arm32-gcc81-O3"}
    by_key: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in data["runs"]:
        if r["binary"] not in targets:
            continue
        cn = r.get("cost_name")
        if cn not in cost_order:
            continue
        by_key[(r["binary"], cn)].append(r["rl"]["mean_resolve_rate"])

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(cost_order))
    w = 0.35

    series = [
        ("gcc_base.arm32-gcc81-O3", "gcc", COLORS["rl"]),
        ("dealII_base.arm32-gcc81-O3", "dealII", "#1565C0"),
    ]

    all_low: List[float] = []
    all_high: List[float] = []
    for i, (full, short, color) in enumerate(series):
        means, stds = [], []
        for cn in cost_order:
            vals = by_key.get((full, cn), [])
            if not vals:
                means.append(float("nan"))
                stds.append(0.0)
            else:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=0)))
        offset = (i - 0.5) * w
        ax.bar(
            x + offset,
            means,
            w,
            yerr=stds,
            capsize=3,
            label=short,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            ecolor="black",
            alpha=0.92,
        )
        for m, s in zip(means, stds):
            if not np.isnan(m):
                all_low.append(m - s)
                all_high.append(m + s)

    ax.set_xticks(x)
    ax.set_xticklabels(cost_order)
    ax.set_ylabel("RL resolve rate")
    ax.set_xlabel("Cost preset")
    ax.set_title("RL sensitivity to cost preset (mean ± std over seeds)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    if all_low and all_high:
        data_min, data_max = min(all_low), max(all_high)
        # Truncate floor at 0.5 when all values stay above it (highlights gcc cost spread).
        y0 = 0.5 if data_min >= 0.5 else max(0.0, data_min - 0.03)
        y1 = min(1.02, data_max + 0.04)
        ax.set_ylim(y0, y1)

    _save_fig(fig, out_path)


def plot_fig3_cross_binary_heatmap_pp(
    sensitivity_path: str,
    cross_path: str,
    out_path: str = "results/figures/cross_binary_heatmap_pp.png",
):
    """3×3 heatmap: RL improvement over best baseline (percentage points)."""
    import matplotlib.pyplot as plt

    sens = _load_json(sensitivity_path)
    cross = _load_json(cross_path)

    n = len(HEATMAP_SHORT_ORDER)
    matrix = np.full((n, n), np.nan)
    # Diagonal from sensitivity baseline runs (same-binary train/eval)
    baseline_runs = [r for r in sens["runs"] if r.get("cost_name") == "baseline"]
    diag_by_short: Dict[str, List[float]] = defaultdict(list)
    for r in baseline_runs:
        full = r["binary"]
        short = HEATMAP_FULL_TO_SHORT.get(full)
        if short is None:
            continue
        rl = r["rl"]["mean_resolve_rate"]
        best_bl = _max_baseline_resolve_rate(r)
        diag_by_short[short].append((rl - best_bl) * 100.0)

    for i, s in enumerate(HEATMAP_SHORT_ORDER):
        if s in diag_by_short:
            matrix[i, i] = np.mean(diag_by_short[s])

    # Off-diagonal from cross-binary JSON (skip diagonal rows in file)
    for exp in cross.get("experiments", []):
        tr = exp["train_binary"]
        ev = exp["eval_binary"]
        ts = HEATMAP_FULL_TO_SHORT.get(tr)
        es = HEATMAP_FULL_TO_SHORT.get(ev)
        if ts is None or es is None or ts == es:
            continue
        imp = exp.get("improvement_over_best")
        if imp is None:
            continue
        i = HEATMAP_SHORT_ORDER.index(ts)
        j = HEATMAP_SHORT_ORDER.index(es)
        matrix[i, j] = float(imp) * 100.0

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    vmax = float(np.nanmax(np.abs(matrix[~np.isnan(matrix)]))) if np.any(~np.isnan(matrix)) else 1.0
    if vmax < 1e-6:
        vmax = 1.0
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#D0D0D0")
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, aspect="equal", vmin=-vmax, vmax=vmax)

    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=11, color="#404040")
            else:
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=11, color="black")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(list(HEATMAP_SHORT_ORDER))
    ax.set_yticklabels(list(HEATMAP_SHORT_ORDER))
    ax.set_xlabel("Eval binary")
    ax.set_ylabel("Train binary")
    ax.set_title("RL vs best baseline (percentage points)")

    fig.colorbar(im, ax=ax, shrink=0.75, label="pp")

    _save_fig(fig, out_path)


def plot_all(
    sensitivity_path: str = "data/calibration/sensitivity_results.json",
    cross_path: str = "data/calibration/cross_binary_results.json",
    figures_dir: str = "results/figures",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": FONT_SIZE})

    os.makedirs(figures_dir, exist_ok=True)
    plot_fig1_baseline_grouped_bars(
        sensitivity_path,
        os.path.join(figures_dir, "sensitivity_baseline_rl_vs_baselines.png"),
    )
    plot_fig2_gcc_dealii_cost_sensitivity(
        sensitivity_path,
        os.path.join(figures_dir, "sensitivity_gcc_dealii_rl_by_cost.png"),
    )
    if os.path.exists(cross_path):
        plot_fig3_cross_binary_heatmap_pp(
            sensitivity_path,
            cross_path,
            os.path.join(figures_dir, "cross_binary_heatmap_pp.png"),
        )
    else:
        print(f"Skip Fig3: missing {cross_path}")


if __name__ == "__main__":
    plot_all()
