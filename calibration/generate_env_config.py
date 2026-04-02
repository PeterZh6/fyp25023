"""Generate RL environment YAML config from calibration outputs.

Combines difficulty distributions, success rates, and cost parameters
into a single YAML config file for the RL environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from calibration.compute_distributions import load_silver_labels


DEFAULT_FALLBACKS = {
    "L2_hard": 0.15,
    "L3_easy": 1.0,
    "L3_medium": 0.95,
    "L3_hard": 0.7,
}


def _fill_null(val: Optional[float], fallback: float) -> float:
    return fallback if val is None else val


def _compute_site_features(data: dict) -> dict:
    """Compute jump_ratio and type_agree_ratio from silver label data."""
    sites = data["sites"]
    if not sites:
        return {"jump_ratio": 0.5, "type_agree_ratio": 0.8}

    n_jump = sum(1 for s in sites if s.get("type") == "jump")
    n_agree = sum(1 for s in sites if s.get("type_agree", False))
    total = len(sites)

    return {
        "jump_ratio": round(n_jump / total, 4),
        "type_agree_ratio": round(n_agree / total, 4),
    }


def generate_config(
    dist_path: str,
    rates_path: str,
    costs_path: str,
    labels_dir: str,
    fallbacks: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if fallbacks is None:
        fallbacks = DEFAULT_FALLBACKS

    with open(dist_path) as f:
        distributions = json.load(f)
    with open(rates_path) as f:
        success_rates = json.load(f)
    with open(costs_path) as f:
        cost_params = json.load(f)

    entries = {name: data for name, data in load_silver_labels(labels_dir)}

    costs = cost_params["costs"]
    for key in ["SKIP", "L1", "L2", "L3"]:
        if costs.get(key) is None:
            defaults = {"SKIP": 0, "L1": 1, "L2": 5, "L3": 20}
            costs[key] = defaults[key]

    estimated_fields: List[str] = []

    per_binary: Dict[str, Any] = {}
    for binary_name, dist_info in distributions["per_binary"].items():
        rates_info = success_rates["per_binary"].get(binary_name, {})
        l1 = rates_info.get("L1", {})
        l2 = rates_info.get("L2", {})
        l3 = rates_info.get("L3", {})

        l2_hard = _fill_null(l2.get("hard"), fallbacks["L2_hard"])
        l3_easy = _fill_null(l3.get("easy"), fallbacks["L3_easy"])
        l3_medium = _fill_null(l3.get("medium"), fallbacks["L3_medium"])
        l3_hard = _fill_null(l3.get("hard"), fallbacks["L3_hard"])

        sr = {
            "SKIP": [0, 0, 0],
            "L1": [
                _fill_null(l1.get("easy"), 1.0),
                _fill_null(l1.get("medium"), 0.5),
                _fill_null(l1.get("hard"), 0.0),
            ],
            "L2": [
                _fill_null(l2.get("easy"), 1.0),
                _fill_null(l2.get("medium"), 0.8),
                l2_hard,
            ],
            "L3": [l3_easy, l3_medium, l3_hard],
        }

        features = {"jump_ratio": 0.5, "type_agree_ratio": 0.8}
        if binary_name in entries:
            features = _compute_site_features(entries[binary_name])

        per_binary[binary_name] = {
            "num_sites": dist_info["total_sites"],
            "difficulty_distribution": [
                dist_info["easy"],
                dist_info["medium"],
                dist_info["hard"],
            ],
            "success_rates": sr,
            "jump_ratio": features["jump_ratio"],
            "type_agree_ratio": features["type_agree_ratio"],
        }

    mixed_no_cpp_dist = distributions["mixed_no_cpp"]
    agg = success_rates.get("aggregate_no_cpp", {})
    l1_agg = agg.get("L1", {})
    l2_agg = agg.get("L2", {})

    mixed_no_cpp = {
        "num_sites": mixed_no_cpp_dist["total_sites"],
        "difficulty_distribution": [
            mixed_no_cpp_dist["easy"],
            mixed_no_cpp_dist["medium"],
            mixed_no_cpp_dist["hard"],
        ],
        "success_rates": {
            "SKIP": [0, 0, 0],
            "L1": [
                _fill_null(l1_agg.get("easy"), 1.0),
                _fill_null(l1_agg.get("medium"), 0.5),
                _fill_null(l1_agg.get("hard"), 0.0),
            ],
            "L2": [
                _fill_null(l2_agg.get("easy"), 1.0),
                _fill_null(l2_agg.get("medium"), 0.8),
                fallbacks["L2_hard"],
            ],
            "L3": [
                fallbacks["L3_easy"],
                fallbacks["L3_medium"],
                fallbacks["L3_hard"],
            ],
        },
        "jump_ratio": 0.5,
        "type_agree_ratio": 0.8,
    }

    estimated_fields.append(f"All binaries: L2.hard (using fallback {fallbacks['L2_hard']})")
    estimated_fields.append(f"All binaries: L3.* (using fallback values)")
    estimated_fields.append(f"All binaries: costs (using manual {costs['L1']}:{costs['L2']}:{costs['L3']})")

    config = {
        "defaults": {
            "costs": costs,
            "success_rate_fallbacks": {
                "L2_hard": fallbacks["L2_hard"],
                "L3_easy": fallbacks["L3_easy"],
                "L3_medium": fallbacks["L3_medium"],
                "L3_hard": fallbacks["L3_hard"],
            },
        },
        "per_binary": per_binary,
        "mixed_no_cpp": mixed_no_cpp,
        "estimated_fields": estimated_fields,
    }

    return config


def write_yaml(config: Dict[str, Any], output_path: str) -> None:
    """Write config as YAML with a header comment."""
    try:
        import yaml
    except ImportError:
        print("[ERROR] PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    header = (
        "# Auto-generated by calibration/generate_env_config.py\n"
        "# Regenerate: python -m calibration.generate_env_config\n\n"
    )
    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RL env config YAML from calibration data"
    )
    parser.add_argument(
        "--distributions", default="data/calibration/difficulty_distributions.json",
    )
    parser.add_argument("--success-rates", default="data/calibration/success_rates.json")
    parser.add_argument("--costs", default="data/calibration/cost_params.json")
    parser.add_argument("--labels-dir", default="data/silver_labels")
    parser.add_argument("--output", default="rl/configs/env_configs.yaml")

    parser.add_argument("--l2-hard-fallback", type=float, default=0.15)
    parser.add_argument("--l3-easy-fallback", type=float, default=1.0)
    parser.add_argument("--l3-medium-fallback", type=float, default=0.95)
    parser.add_argument("--l3-hard-fallback", type=float, default=0.7)
    args = parser.parse_args()

    fallbacks = {
        "L2_hard": args.l2_hard_fallback,
        "L3_easy": args.l3_easy_fallback,
        "L3_medium": args.l3_medium_fallback,
        "L3_hard": args.l3_hard_fallback,
    }

    config = generate_config(
        args.distributions,
        args.success_rates,
        args.costs,
        args.labels_dir,
        fallbacks=fallbacks,
    )

    write_yaml(config, args.output)

    print("\n=== Estimated fields ===")
    for field in config["estimated_fields"]:
        print(f"  - {field}")


if __name__ == "__main__":
    main()
