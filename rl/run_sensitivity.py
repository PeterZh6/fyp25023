"""Cost-ratio sensitivity: sweep L2/L3 presets per binary, train PPO + baseline eval.

Reuses rl.train (train_single, evaluate_trained_model) and TrainConfig.cost_overrides.

Run from repository root: python -m rl.run_sensitivity [args]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Dict, List

import numpy as np

from calibration.config import filter_binary_names
from rl.baselines import ALL_BASELINES
from rl.train import TrainConfig, evaluate_trained_model, train_single

# L1 fixed at 1; L2/L3 match common ratio tiers
COST_PRESETS: Dict[str, Dict[str, float]] = {
    "baseline": {"L1": 1.0, "L2": 5.0, "L3": 20.0},
    "low": {"L1": 1.0, "L2": 3.0, "L3": 15.0},
    "high": {"L1": 1.0, "L2": 10.0, "L3": 50.0},
}

DEFAULT_BINARIES: List[str] = [
    "bzip2_base.arm32-gcc81-O3",
    "openssl",
    "ssh",
    "gcc_base.arm32-gcc81-O3",
    "dealII_base.arm32-gcc81-O3",
]

RESULT_JSON = "data/calibration/sensitivity_results.json"


def _strategy_columns() -> List[str]:
    return ["RL"] + list(ALL_BASELINES.keys())


def _print_flat_binary_cost_summary(
    summary: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """One table: each row = (binary, cost preset) with all strategy resolve rates."""
    cols = _strategy_columns()
    w_bin = max(8, max(len(b) for b in summary) if summary else 8)
    w_cost = max(len("cost"), max(len(c) for b in summary for c in summary[b]) if summary else 8)
    w_cell = 8
    header = (
        f"{'binary':<{w_bin}s}"
        f"{'cost':<{w_cost}s}"
        + "".join(f"{c:>{w_cell}s}" for c in cols)
    )
    print(header)
    print("-" * len(header))
    for binary in summary:
        for cost_name in COST_PRESETS:
            if cost_name not in summary[binary]:
                continue
            r = summary[binary][cost_name]
            cells = "".join(f"{r.get(c, float('nan')):>{w_cell}.4f}" for c in cols)
            print(f"{binary:<{w_bin}s}{cost_name:<{w_cost}s}{cells}")


def _print_cost_x_strategy_table(
    title: str,
    rows: Dict[str, Dict[str, float]],
) -> None:
    """rows: cost_preset -> {strategy -> mean resolve rate}"""
    cols = _strategy_columns()
    w_cost = max(len("cost"), max(len(k) for k in rows))
    w_cell = 8

    header = f"{ 'cost':<{w_cost}s}" + "".join(f"{c:>{w_cell}s}" for c in cols)
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for cost_name in COST_PRESETS:
        if cost_name not in rows:
            continue
        r = rows[cost_name]
        cells = "".join(f"{r.get(c, float('nan')):>{w_cell}.4f}" for c in cols)
        print(f"{cost_name:<{w_cost}s}{cells}")


def _run_cost_group(
    train_cfg: TrainConfig,
    binary: str,
    cost_name: str,
    cost_overrides: Dict[str, float],
    seeds: List[int],
    eval_episodes: int,
    group_save_dir: str,
) -> Dict[str, Any]:
    rl_per_seed: List[float] = []
    baseline_per_seed: Dict[str, List[float]] = {n: [] for n in ALL_BASELINES}

    for seed in seeds:
        run_save = os.path.join(group_save_dir, f"seed_{seed}")
        cfg = dataclasses.replace(
            train_cfg,
            binary_name=binary,
            cost_overrides=cost_overrides,
            save_dir=run_save,
        )
        print(f"    train seed={seed} ...")
        trained = train_single(cfg, seed=seed)
        ev = evaluate_trained_model(
            trained["best_model_path"],
            cfg,
            n_episodes=eval_episodes,
            seed=seed,
        )
        rl_per_seed.append(float(ev["rl"]["mean_resolve_rate"]))
        for name in ALL_BASELINES:
            baseline_per_seed[name].append(
                float(ev["baselines"][name]["mean_resolve_rate"])
            )

    means: Dict[str, float] = {
        "RL": float(np.mean(rl_per_seed)),
    }
    means.update(
        {n: float(np.mean(baseline_per_seed[n])) for n in ALL_BASELINES}
    )
    stds: Dict[str, float] = {
        "RL": float(np.std(rl_per_seed)),
    }
    stds.update(
        {n: float(np.std(baseline_per_seed[n])) for n in ALL_BASELINES}
    )

    return {
        "cost_name": cost_name,
        "cost_overrides": cost_overrides,
        "mean_resolve_rate": means,
        "std_resolve_rate": stds,
        "per_seed": {
            "RL": rl_per_seed,
            **{n: baseline_per_seed[n] for n in ALL_BASELINES},
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost-ratio sensitivity (PPO + baselines)")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument(
        "--binary",
        nargs="+",
        default=DEFAULT_BINARIES,
        help="One or more binary keys (YAML names or short names)",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--config", default="rl/configs/env_configs.yaml")
    p.add_argument("--budget-ratio", type=float, default=2.0)
    p.add_argument("--cost-lambda", type=float, default=0.02)
    p.add_argument("--save-dir", default="results/sensitivity")
    p.add_argument("--output", default=RESULT_JSON)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    binaries = filter_binary_names(args.binary)
    if not binaries:
        raise SystemExit("No binaries left after filtering (check names / EXCLUDED_BINARIES).")

    train_cfg = TrainConfig(
        env_config_path=args.config,
        budget_ratio=args.budget_ratio,
        total_timesteps=args.timesteps,
        cost_lambda=args.cost_lambda,
        num_seeds=len(args.seeds),
    )

    all_results: Dict[str, Any] = {
        "meta": {
            "timesteps": args.timesteps,
            "seeds": list(args.seeds),
            "eval_episodes": args.eval_episodes,
            "binaries": binaries,
            "cost_presets": COST_PRESETS,
        },
        "by_binary": {},
    }

    summary_for_print: Dict[str, Dict[str, Dict[str, float]]] = {}

    for binary in binaries:
        print(f"\n{'=' * 60}\nBinary: {binary}\n{'=' * 60}")
        per_cost: Dict[str, Any] = {}
        table_rows: Dict[str, Dict[str, float]] = {}

        for cost_name, overrides in COST_PRESETS.items():
            print(f"\n--- cost preset: {cost_name} {overrides} ---")
            group_dir = os.path.join(
                args.save_dir,
                binary.replace(os.sep, "_"),
                cost_name,
            )
            block = _run_cost_group(
                train_cfg,
                binary,
                cost_name,
                overrides,
                args.seeds,
                args.eval_episodes,
                group_dir,
            )
            per_cost[cost_name] = block
            table_rows[cost_name] = block["mean_resolve_rate"]

        all_results["by_binary"][binary] = per_cost
        summary_for_print[binary] = table_rows
        _print_cost_x_strategy_table(
            f"Resolve rate (mean over seeds {args.seeds}): {binary}",
            table_rows,
        )

    print(f"\n{'=' * 60}\nSUMMARY: binary × cost × strategy → mean resolve rate\n{'=' * 60}")
    _print_flat_binary_cost_summary(summary_for_print)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
