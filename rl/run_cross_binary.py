"""Cross-binary generalization: evaluate PPO trained on binary A on binary B's env.

Loads models from data/calibration/models/ (same naming as run_sensitivity).
Merges same-binary diagonal rows from sensitivity_results.json so one JSON
suffices for heatmaps.

Run from repository root: python -m rl.run_cross_binary [args]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rl.run_sensitivity import COST_PRESETS, calibration_model_path
from rl.train import TrainConfig, evaluate_trained_model

# Must stay in sync with rl.run_sensitivity parse_args --eval-episodes default.
DEFAULT_EVAL_EPISODES = 200

RESULT_JSON = "data/calibration/cross_binary_results.json"
SENSITIVITY_JSON = "data/calibration/sensitivity_results.json"
COST_NAME = "baseline"

GCC = "gcc_base.arm32-gcc81-O3"
DEALII = "dealII_base.arm32-gcc81-O3"
SSH = "ssh"

# (train_binary, eval_binary); per_binary key for OpenSSH is "ssh", not ssh_base...
DEFAULT_EXPERIMENTS: List[Tuple[str, str]] = [
    (GCC, SSH),
    (GCC, DEALII),
    (DEALII, GCC),
]


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _best_baseline_from_metrics(
    baselines: Dict[str, Any],
) -> Tuple[Optional[str], Optional[float]]:
    if not baselines:
        return None, None
    best = max(baselines, key=lambda k: float(baselines[k]["mean_resolve_rate"]))
    return best, float(baselines[best]["mean_resolve_rate"])


def _load_sensitivity_runs(path: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not os.path.isfile(path):
        return None, []
    with open(path) as f:
        doc = json.load(f)
    runs = doc.get("runs")
    if not isinstance(runs, list):
        return doc.get("meta") or {}, []
    return doc.get("meta") or {}, runs


def _find_sensitivity_run(
    runs: List[Dict[str, Any]],
    binary: str,
    cost_name: str,
    seed: int,
) -> Optional[Dict[str, Any]]:
    match: Optional[Dict[str, Any]] = None
    for r in runs:
        if not isinstance(r, dict):
            continue
        if (
            r.get("binary") == binary
            and r.get("cost_name") == cost_name
            and r.get("seed") == seed
        ):
            match = r
    return match


def _diagonal_record_from_sensitivity(
    binary: str,
    run: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    rl = run.get("rl") or {}
    baselines = run.get("baselines") or {}
    best_name, best_rate = _best_baseline_from_metrics(baselines)
    model_path = calibration_model_path(binary, COST_NAME, seed)
    rec: Dict[str, Any] = {
        "train_binary": binary,
        "eval_binary": binary,
        "cost_name": COST_NAME,
        "seed": seed,
        "source": "sensitivity_results",
        "model_path": model_path if os.path.isfile(model_path) else None,
        "rl_eval": dict(rl) if isinstance(rl, dict) else rl,
        "baselines": dict(baselines) if isinstance(baselines, dict) else baselines,
        "best_baseline_name": best_name,
        "best_baseline_resolve_rate": best_rate,
    }
    if isinstance(rl, dict) and best_rate is not None:
        rec["improvement_over_best"] = float(rl.get("mean_resolve_rate", 0.0)) - best_rate
    return rec


def _cross_eval_record(
    train_binary: str,
    eval_binary: str,
    model_path: str,
    seed: int,
    ev: Dict[str, Any],
) -> Dict[str, Any]:
    rl = ev.get("rl") or {}
    baselines = ev.get("baselines") or {}
    best_name, best_rate = _best_baseline_from_metrics(baselines)
    return {
        "train_binary": train_binary,
        "eval_binary": eval_binary,
        "cost_name": COST_NAME,
        "seed": seed,
        "source": "cross_eval",
        "model_path": model_path,
        "rl_eval": dict(rl) if isinstance(rl, dict) else rl,
        "baselines": dict(baselines) if isinstance(baselines, dict) else baselines,
        "best_baseline_name": best_name,
        "best_baseline_resolve_rate": best_rate,
        "improvement_over_best": ev.get("improvement_over_best"),
    }


def _unique_binaries(pairs: Sequence[Tuple[str, str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for a, b in pairs:
        for x in (a, b):
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-binary PPO eval + sensitivity diagonal merge")
    p.add_argument(
        "--output",
        default=RESULT_JSON,
        help=f"Output JSON (default: {RESULT_JSON})",
    )
    p.add_argument(
        "--sensitivity-results",
        default=SENSITIVITY_JSON,
        help=f"Sensitivity runs JSON for diagonal (default: {SENSITIVITY_JSON})",
    )
    p.add_argument(
        "--no-diagonal",
        action="store_true",
        help="Do not load diagonal rows from sensitivity JSON",
    )
    p.add_argument("--config", default="rl/configs/env_configs.yaml")
    p.add_argument("--budget-ratio", type=float, default=2.0)
    p.add_argument("--cost-lambda", type=float, default=0.02)
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=f"Must match run_sensitivity default ({DEFAULT_EVAL_EPISODES}) for fair comparison",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="RL-only eval (faster); baselines fields empty",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiments: List[Dict[str, Any]] = []
    missing_diagonals: List[Dict[str, str]] = []

    train_cfg = TrainConfig(
        env_config_path=args.config,
        budget_ratio=args.budget_ratio,
        total_timesteps=200_000,
        cost_lambda=args.cost_lambda,
        num_seeds=1,
        cost_overrides=COST_PRESETS[COST_NAME],
    )

    sens_meta: Optional[Dict[str, Any]] = None
    sens_runs: List[Dict[str, Any]] = []
    if not args.no_diagonal:
        sens_meta, sens_runs = _load_sensitivity_runs(args.sensitivity_results)
        if sens_meta and sens_meta.get("eval_episodes") not in (None, args.eval_episodes):
            warnings.warn(
                f"sensitivity meta eval_episodes={sens_meta.get('eval_episodes')!r} "
                f"!= --eval-episodes {args.eval_episodes}; diagonal vs cross noise may differ",
                stacklevel=1,
            )

    for train_binary, eval_binary in DEFAULT_EXPERIMENTS:
        if train_binary == eval_binary:
            continue
        model_path = calibration_model_path(train_binary, COST_NAME, args.seed)
        if not os.path.isfile(model_path):
            raise SystemExit(
                f"Missing model for cross-eval: {model_path}\n"
                f"Train with run_sensitivity for {train_binary!r} cost {COST_NAME} seed {args.seed}."
            )

        cfg = dataclasses.replace(
            train_cfg,
            binary_name=train_binary,
        )
        print(f"\n{'=' * 60}\ntrain={train_binary} eval={eval_binary}\n{'=' * 60}")
        ev = evaluate_trained_model(
            model_path,
            cfg,
            n_episodes=args.eval_episodes,
            seed=args.seed,
            test_binary=eval_binary,
            skip_baselines=args.skip_baselines,
        )
        experiments.append(
            _cross_eval_record(train_binary, eval_binary, model_path, args.seed, ev)
        )
        rr = ev["rl"]["mean_resolve_rate"]
        print(f"  RL mean_resolve_rate={rr:.4f}")

    if not args.no_diagonal:
        binaries_for_diag = _unique_binaries(DEFAULT_EXPERIMENTS)
        if sens_meta is None and not sens_runs:
            warnings.warn(
                f"No sensitivity file or empty runs: {args.sensitivity_results!r}; "
                "skipping all diagonal rows",
                stacklevel=1,
            )
            for b in binaries_for_diag:
                missing_diagonals.append(
                    {"binary": b, "reason": "no_sensitivity_file_or_empty_runs"}
                )
        else:
            for b in sorted(binaries_for_diag):
                run = _find_sensitivity_run(sens_runs, b, COST_NAME, args.seed)
                if run is None:
                    warnings.warn(
                        f"No sensitivity run for diagonal binary={b!r} "
                        f"cost={COST_NAME!r} seed={args.seed}; skipping",
                        stacklevel=1,
                    )
                    missing_diagonals.append({"binary": b, "reason": "no_matching_run"})
                    continue
                experiments.append(_diagonal_record_from_sensitivity(b, run, args.seed))
                print(f"  [diagonal] {b} ← sensitivity_results")

    meta = {
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "cost_name": COST_NAME,
        "config_path": args.config,
        "budget_ratio": args.budget_ratio,
        "cost_lambda": args.cost_lambda,
        "sensitivity_results_path": args.sensitivity_results,
        "diagonal_enabled": not args.no_diagonal,
        "skip_baselines": args.skip_baselines,
        "missing_diagonals": missing_diagonals,
        "experiment_pairs": [list(t) for t in DEFAULT_EXPERIMENTS],
    }

    doc = {"meta": meta, "experiments": experiments}
    _atomic_write_json(args.output, doc)
    print(f"\nWrote {args.output} ({len(experiments)} experiments)")


if __name__ == "__main__":
    main()
