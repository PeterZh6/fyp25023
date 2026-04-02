"""Compute success rate matrix from silver label data.

For each (analysis level, difficulty) pair, compute the probability that the
analysis tool successfully resolves the indirect branch target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from calibration.compute_distributions import load_silver_labels


def _compute_per_binary_rates(
    data: dict,
) -> tuple[Dict[str, Dict[str, Optional[float]]], Dict[str, int], Dict[str, Any]]:
    """Compute L1/L2 success rates for a single binary.

    Returns (success_rates, site_counts, medium_breakdown).
    """
    sites = data["sites"]

    counts: Dict[str, Dict[str, int]] = {
        d: {"total": 0, "l1_success": 0, "l2_success": 0}
        for d in ["easy", "medium", "hard"]
    }
    medium_ghidra_only = 0
    medium_angr_only = 0
    medium_both_partial = 0
    medium_conflict = 0
    medium_partial = 0

    for site in sites:
        label = site["label"]
        if label not in counts:
            continue
        counts[label]["total"] += 1

        ghidra_ok = site.get("ghidra_found", False) and len(site.get("ghidra_targets", [])) > 0
        angr_ok = site.get("angr_found", False) and len(site.get("angr_targets", [])) > 0

        if ghidra_ok:
            counts[label]["l1_success"] += 1
        if angr_ok:
            counts[label]["l2_success"] += 1

        if label == "medium":
            detail = site.get("label_detail", "")
            if detail == "medium_conflict":
                medium_conflict += 1
            elif detail == "medium_partial":
                medium_partial += 1

            if ghidra_ok and not angr_ok:
                medium_ghidra_only += 1
            elif not ghidra_ok and angr_ok:
                medium_angr_only += 1
            elif ghidra_ok and angr_ok:
                medium_both_partial += 1

    def _rate(successes: int, total: int) -> Optional[float]:
        if total == 0:
            return None
        return round(successes / total, 4)

    success_rates = {
        "L1": {
            "easy": _rate(counts["easy"]["l1_success"], counts["easy"]["total"]),
            "medium": _rate(counts["medium"]["l1_success"], counts["medium"]["total"]),
            "hard": _rate(counts["hard"]["l1_success"], counts["hard"]["total"]),
        },
        "L2": {
            "easy": _rate(counts["easy"]["l2_success"], counts["easy"]["total"]),
            "medium": _rate(counts["medium"]["l2_success"], counts["medium"]["total"]),
            "hard": None,  # no ground truth for L2 on hard
        },
        "L3": {
            "easy": None,
            "medium": None,
            "hard": None,
        },
    }

    site_counts = {
        "easy": counts["easy"]["total"],
        "medium": counts["medium"]["total"],
        "hard": counts["hard"]["total"],
    }

    medium_breakdown = {
        "medium_conflict": medium_conflict,
        "medium_partial": medium_partial,
        "ghidra_only": medium_ghidra_only,
        "angr_only": medium_angr_only,
        "both_partial": medium_both_partial,
    }

    return success_rates, site_counts, medium_breakdown


def _compute_aggregate(
    all_rates: Dict[str, Any],
    exclude_cpp: bool = False,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute aggregate success rates across binaries."""
    cpp_keywords = ["xalan", "dealii"]
    agg_counts: Dict[str, Dict[str, Dict[str, int]]] = {
        level: {d: {"success": 0, "total": 0} for d in ["easy", "medium", "hard"]}
        for level in ["L1", "L2"]
    }

    for binary_name, info in all_rates.items():
        if exclude_cpp and any(kw in binary_name.lower() for kw in cpp_keywords):
            continue
        site_counts = info["site_counts"]

        for level in ["L1", "L2"]:
            for diff in ["easy", "medium", "hard"]:
                n = site_counts[diff]
                rate = info[level][diff]
                if rate is not None and n > 0:
                    agg_counts[level][diff]["success"] += int(round(rate * n))
                    agg_counts[level][diff]["total"] += n

    result: Dict[str, Dict[str, Optional[float]]] = {}
    for level in ["L1", "L2", "L3"]:
        result[level] = {}
        for diff in ["easy", "medium", "hard"]:
            if level == "L3":
                result[level][diff] = None
            elif agg_counts[level][diff]["total"] > 0:
                result[level][diff] = round(
                    agg_counts[level][diff]["success"]
                    / agg_counts[level][diff]["total"],
                    4,
                )
            else:
                result[level][diff] = None
    return result


def compute_success_rates(
    input_dir: str,
    l2_hard_estimate: Optional[float] = None,
    l3_easy: Optional[float] = None,
    l3_medium: Optional[float] = None,
    l3_hard: Optional[float] = None,
) -> Dict[str, Any]:
    entries = load_silver_labels(input_dir)

    per_binary: Dict[str, Any] = {}
    medium_breakdown: Dict[str, Any] = {}

    for binary_name, data in entries:
        rates, site_counts, med_bkdn = _compute_per_binary_rates(data)

        estimated_fields = []
        if l2_hard_estimate is not None:
            rates["L2"]["hard"] = l2_hard_estimate
            estimated_fields.append("L2.hard")
        if l3_easy is not None:
            rates["L3"]["easy"] = l3_easy
            estimated_fields.append("L3.easy")
        if l3_medium is not None:
            rates["L3"]["medium"] = l3_medium
            estimated_fields.append("L3.medium")
        if l3_hard is not None:
            rates["L3"]["hard"] = l3_hard
            estimated_fields.append("L3.hard")
        if not estimated_fields:
            estimated_fields = ["L2.hard", "L3.*"]

        per_binary[binary_name] = {
            "L1": rates["L1"],
            "L2": rates["L2"],
            "L3": rates["L3"],
            "site_counts": site_counts,
            "estimated_fields": estimated_fields,
        }
        medium_breakdown[binary_name] = med_bkdn

    aggregate_all = _compute_aggregate(per_binary, exclude_cpp=False)
    aggregate_no_cpp = _compute_aggregate(per_binary, exclude_cpp=True)

    return {
        "per_binary": per_binary,
        "aggregate_all": aggregate_all,
        "aggregate_no_cpp": aggregate_no_cpp,
        "medium_breakdown": medium_breakdown,
    }


def print_summary(result: Dict[str, Any]) -> None:
    print("\n=== Aggregate Success Rates (all binaries) ===")
    agg = result["aggregate_all"]
    print(f"  {'Level':<5s} {'Easy':>8s} {'Medium':>8s} {'Hard':>8s}")
    for level in ["L1", "L2", "L3"]:
        vals = []
        for d in ["easy", "medium", "hard"]:
            v = agg[level][d]
            vals.append(f"{v*100:.1f}%" if v is not None else "N/A")
        print(f"  {level:<5s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s}")

    print("\n=== Aggregate (excluding C++) ===")
    agg2 = result["aggregate_no_cpp"]
    print(f"  {'Level':<5s} {'Easy':>8s} {'Medium':>8s} {'Hard':>8s}")
    for level in ["L1", "L2", "L3"]:
        vals = []
        for d in ["easy", "medium", "hard"]:
            v = agg2[level][d]
            vals.append(f"{v*100:.1f}%" if v is not None else "N/A")
        print(f"  {level:<5s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s}")

    print("\n=== Medium Breakdown (top 5 by total medium) ===")
    bkdn = result["medium_breakdown"]
    rows = sorted(bkdn.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]
    for name, info in rows:
        total_med = info["medium_conflict"] + info["medium_partial"]
        print(
            f"  {name}: total={total_med}, conflict={info['medium_conflict']}, "
            f"partial={info['medium_partial']}, "
            f"ghidra_only={info['ghidra_only']}, angr_only={info['angr_only']}, "
            f"both_partial={info['both_partial']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute success rates from silver labels"
    )
    parser.add_argument("--input-dir", default="data/silver_labels")
    parser.add_argument("--output", default="data/calibration/success_rates.json")
    parser.add_argument("--l2-hard-estimate", type=float, default=None)
    parser.add_argument("--l3-easy", type=float, default=None)
    parser.add_argument("--l3-medium", type=float, default=None)
    parser.add_argument("--l3-hard", type=float, default=None)
    args = parser.parse_args()

    result = compute_success_rates(
        args.input_dir,
        l2_hard_estimate=args.l2_hard_estimate,
        l3_easy=args.l3_easy,
        l3_medium=args.l3_medium,
        l3_hard=args.l3_hard,
    )
    print_summary(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to: {args.output}")


if __name__ == "__main__":
    main()
