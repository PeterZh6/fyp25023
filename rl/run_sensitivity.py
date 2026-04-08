"""Cost-ratio sensitivity: sweep L2/L3 presets per binary, train PPO + baseline eval.

Reuses rl.train (train_single, evaluate_trained_model) and TrainConfig.cost_overrides.

Run from repository root: python -m rl.run_sensitivity [args]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

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
CALIBRATION_MODEL_DIR = "data/calibration/models"


def calibration_model_path(binary: str, cost_name: str, seed: int) -> str:
    """SB3 zip path used for sensitivity training export and --eval-only load."""
    safe = binary.replace(os.sep, "_")
    return os.path.join(CALIBRATION_MODEL_DIR, f"{safe}_{cost_name}_{seed}.zip")


def _experiment_dedupe_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        r["binary"],
        r["cost_name"],
        r["seed"],
        r.get("timesteps"),
        r.get("deterministic_reward", False),
        r.get("sunk_cost_l1", False),
    )


def _dedupe_runs_last_wins(runs: List[Dict[str, Any]]) -> None:
    """One row per (binary, cost, seed, timesteps); last wins."""
    merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for r in runs:
        merged[_experiment_dedupe_key(r)] = r
    runs[:] = list(merged.values())


def _strategy_columns(skip_baselines: bool) -> List[str]:
    if skip_baselines:
        return ["RL"]
    return ["RL"] + list(ALL_BASELINES.keys())


def _resolve_cost_presets(cost_arg: Optional[List[str]]) -> List[Tuple[str, Dict[str, float]]]:
    if not cost_arg:
        return list(COST_PRESETS.items())
    bad = [n for n in cost_arg if n not in COST_PRESETS]
    if bad:
        raise SystemExit(
            f"Unknown --cost {bad!r}; choose from {list(COST_PRESETS.keys())}"
        )
    return [(n, COST_PRESETS[n]) for n in cost_arg]


def _run_fingerprint(
    binary: str,
    cost_name: str,
    seed: int,
    timesteps: int,
    deterministic_reward: bool = False,
    sunk_cost_l1: bool = False,
) -> Tuple[Any, ...]:
    """Training resume key only (eval / baselines flags are not part of it)."""
    return (binary, cost_name, seed, timesteps, deterministic_reward, sunk_cost_l1)


def _fp_from_run(r: Dict[str, Any], default_timesteps: int) -> Tuple[Any, ...]:
    return _run_fingerprint(
        r["binary"],
        r["cost_name"],
        r["seed"],
        r.get("timesteps", default_timesteps),
        r.get("deterministic_reward", False),
        r.get("sunk_cost_l1", False),
    )


def _load_resume_json(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not os.path.isfile(path):
        return {}, []
    with open(path) as f:
        doc = json.load(f)
    runs = doc.get("runs")
    if not isinstance(runs, list):
        return doc.get("meta") or {}, []
    return doc.get("meta") or {}, runs


def _aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """by_binary[binary][cost_name] -> same shape as before (mean/std/per_seed)."""
    # (binary, cost_name) -> lists
    rl_lists: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    bl_lists: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cost_overrides_map: Dict[Tuple[str, str], Dict[str, float]] = {}

    for run in runs:
        b = run["binary"]
        c = run["cost_name"]
        key = (b, c)
        cost_overrides_map[key] = run.get("cost_overrides") or COST_PRESETS.get(c, {})
        rl_lists[key].append(float(run["rl"]["mean_resolve_rate"]))
        baselines = run.get("baselines") or {}
        for name, metrics in baselines.items():
            if isinstance(metrics, dict) and "mean_resolve_rate" in metrics:
                bl_lists[key][name].append(float(metrics["mean_resolve_rate"]))

    by_binary: Dict[str, Any] = {}
    for (b, c), rl_vals in rl_lists.items():
        key = (b, c)
        means: Dict[str, float] = {"RL": float(np.mean(rl_vals))}
        stds: Dict[str, float] = {"RL": float(np.std(rl_vals))}
        per_seed: Dict[str, List[float]] = {"RL": list(rl_vals)}

        for name in ALL_BASELINES:
            vals = bl_lists[key].get(name, [])
            if vals:
                means[name] = float(np.mean(vals))
                stds[name] = float(np.std(vals))
                per_seed[name] = list(vals)

        if b not in by_binary:
            by_binary[b] = {}
        by_binary[b][c] = {
            "cost_name": c,
            "cost_overrides": cost_overrides_map[key],
            "mean_resolve_rate": means,
            "std_resolve_rate": stds,
            "per_seed": per_seed,
        }

    return by_binary


def _build_summary_for_print(
    by_binary: Dict[str, Any], cost_order: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for binary, per_cost in by_binary.items():
        out[binary] = {}
        for cn in cost_order:
            if cn in per_cost:
                out[binary][cn] = per_cost[cn]["mean_resolve_rate"]
    return out


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _print_flat_binary_cost_summary(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    cost_order: List[str],
    skip_baselines: bool,
) -> None:
    cols = _strategy_columns(skip_baselines)
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
        for cost_name in cost_order:
            if cost_name not in summary[binary]:
                continue
            r = summary[binary][cost_name]
            cells = "".join(f"{r.get(c, float('nan')):>{w_cell}.4f}" for c in cols)
            print(f"{binary:<{w_bin}s}{cost_name:<{w_cost}s}{cells}")


def _print_cost_x_strategy_table(
    title: str,
    rows: Dict[str, Dict[str, float]],
    cost_order: List[str],
    skip_baselines: bool,
) -> None:
    cols = _strategy_columns(skip_baselines)
    w_cost = max(len("cost"), max(len(k) for k in rows) if rows else len("cost"))
    w_cell = 8

    header = f"{'cost':<{w_cost}s}" + "".join(f"{c:>{w_cell}s}" for c in cols)
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for cost_name in cost_order:
        if cost_name not in rows:
            continue
        r = rows[cost_name]
        cells = "".join(f"{r.get(c, float('nan')):>{w_cell}.4f}" for c in cols)
        print(f"{cost_name:<{w_cost}s}{cells}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost-ratio sensitivity (PPO + baselines)")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument(
        "--binary",
        nargs="+",
        default=DEFAULT_BINARIES,
        help="One or more binary keys (YAML names or short names)",
    )
    p.add_argument(
        "--cost",
        nargs="+",
        default=None,
        metavar="PRESET",
        help=f"Subset of cost presets (default: all). Choices: {list(COST_PRESETS.keys())}",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--config", default="rl/configs/env_configs.yaml")
    p.add_argument("--budget-ratio", type=float, default=2.0)
    p.add_argument("--cost-lambda", type=float, default=0.02)
    p.add_argument("--save-dir", default="results/sensitivity")
    p.add_argument("--output", default=RESULT_JSON)
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Only evaluate RL after training; skip baseline policies",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Load --output; skip re-training when (binary,cost,seed,timesteps) already done; still re-runs eval",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Never train; load PPO zip from data/calibration/models/ and run eval (--skip-baselines still applies)",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic L1/L2 reward (pre-sampled per episode)",
    )
    p.add_argument(
        "--sunk-cost",
        action="store_true",
        help="Enable sunk-cost L1 mode (L1 pre-applied at reset; agent sees SKIP/L2/L3 only)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    binaries = filter_binary_names(args.binary)
    if not binaries:
        raise SystemExit("No binaries left after filtering (check names / EXCLUDED_BINARIES).")

    cost_items = _resolve_cost_presets(args.cost)
    cost_order = [name for name, _ in cost_items]

    if args.deterministic and args.output == RESULT_JSON:
        args.output = "data/calibration/deterministic_results.json"
    if args.sunk_cost and args.output == RESULT_JSON:
        args.output = "data/calibration/sunk_cost_results.json"

    train_cfg = TrainConfig(
        env_config_path=args.config,
        budget_ratio=args.budget_ratio,
        total_timesteps=args.timesteps,
        cost_lambda=args.cost_lambda,
        num_seeds=len(args.seeds),
        deterministic_reward=args.deterministic,
        sunk_cost_l1=args.sunk_cost,
    )

    prev_meta: Dict[str, Any] = {}
    runs: List[Dict[str, Any]] = []
    completed: Set[Tuple[Any, ...]] = set()

    if args.resume and os.path.isfile(args.output):
        prev_meta, runs = _load_resume_json(args.output)
        _dedupe_runs_last_wins(runs)
        completed = {_fp_from_run(r, args.timesteps) for r in runs}
        print(f"[resume] Loaded {len(runs)} runs; {len(completed)} trained keys from {args.output}")

    meta = {
        **prev_meta,
        "timesteps": args.timesteps,
        "seeds": list(args.seeds),
        "eval_episodes": args.eval_episodes,
        "binaries": binaries,
        "cost_presets": {k: v for k, v in COST_PRESETS.items()},
        "costs_run": cost_order,
        "skip_baselines": args.skip_baselines,
        "resume": args.resume,
        "eval_only": args.eval_only,
        "deterministic_reward": args.deterministic,
        "sunk_cost_l1": args.sunk_cost,
    }

    def save_document() -> None:
        _dedupe_runs_last_wins(runs)
        by_binary = _aggregate_runs(runs)
        doc = {
            "meta": meta,
            "runs": runs,
            "by_binary": by_binary,
        }
        _atomic_write_json(args.output, doc)

    table_skip_baselines = args.skip_baselines

    for binary in binaries:
        print(f"\n{'=' * 60}\nBinary: {binary}\n{'=' * 60}")

        for cost_name, overrides in cost_items:
            print(f"\n--- cost preset: {cost_name} {overrides} ---")
            group_dir = os.path.join(
                args.save_dir,
                binary.replace(os.sep, "_"),
                cost_name,
            )

            for seed in args.seeds:
                fp = _run_fingerprint(binary, cost_name, seed, args.timesteps, args.deterministic, args.sunk_cost)
                run_save = os.path.join(group_dir, f"seed_{seed}")
                cal_model_zip = calibration_model_path(binary, cost_name, seed)
                cfg = dataclasses.replace(
                    train_cfg,
                    binary_name=binary,
                    cost_overrides=overrides,
                    save_dir=run_save,
                    sensitivity_cost_name=None if args.eval_only else cost_name,
                )

                if args.eval_only:
                    if not os.path.isfile(cal_model_zip):
                        raise SystemExit(
                            f"Missing model for eval-only: {cal_model_zip}\n"
                            "Train first (without --eval-only) so train_single saves there."
                        )
                    print(f"    eval-only seed={seed} ← {cal_model_zip}")
                else:
                    skip_train = (
                        args.resume
                        and fp in completed
                        and os.path.isfile(cal_model_zip)
                    )
                    if skip_train:
                        print(f"    skip training (resume) seed={seed}")
                    else:
                        print(f"    train seed={seed} ...")
                        train_single(cfg, seed=seed)

                ev = evaluate_trained_model(
                    cal_model_zip,
                    cfg,
                    n_episodes=args.eval_episodes,
                    seed=seed,
                    skip_baselines=args.skip_baselines,
                )

                run_record: Dict[str, Any] = {
                    "binary": binary,
                    "cost_name": cost_name,
                    "cost_overrides": overrides,
                    "seed": seed,
                    "timesteps": args.timesteps,
                    "deterministic_reward": args.deterministic,
                    "sunk_cost_l1": args.sunk_cost,
                    "skip_baselines": args.skip_baselines,
                    "eval_episodes": args.eval_episodes,
                    "eval_only": args.eval_only,
                    "rl": ev["rl"],
                    "baselines": ev.get("baselines") or {},
                }
                runs.append(run_record)
                _dedupe_runs_last_wins(runs)
                completed = {_fp_from_run(r, args.timesteps) for r in runs}
                save_document()
                print(f"    saved → {args.output} ({len(runs)} runs)")

        # Per-binary table from full file state (includes resume from other binaries)
        by_binary_now = _aggregate_runs(runs)
        if binary in by_binary_now:
            per_binary_summary = {
                cn: by_binary_now[binary][cn]["mean_resolve_rate"]
                for cn in cost_order
                if cn in by_binary_now[binary]
            }
            _print_cost_x_strategy_table(
                f"Resolve rate (mean over seeds {args.seeds}): {binary}",
                per_binary_summary,
                cost_order,
                table_skip_baselines,
            )

    print(f"\n{'=' * 60}\nSUMMARY: binary × cost × strategy → mean resolve rate\n{'=' * 60}")
    by_final = _aggregate_runs(runs)
    summary_final = _build_summary_for_print(by_final, cost_order)
    # Only print rows for binaries in this invocation and costs we care about
    summary_filtered = {
        b: {c: summary_final[b][c] for c in cost_order if c in summary_final.get(b, {})}
        for b in binaries
        if b in summary_final
    }
    _print_flat_binary_cost_summary(
        summary_filtered, cost_order, table_skip_baselines
    )

    save_document()
    print(f"\nWrote {args.output} ({len(runs)} runs total)")


if __name__ == "__main__":
    main()
