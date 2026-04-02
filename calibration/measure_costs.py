"""Measure or manually specify analysis cost ratios.

Two modes:
  --manual: directly specify L1/L2/L3 cost ratios
  --from-logs: parse timing logs to compute amortized per-site costs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from calibration.compute_distributions import load_silver_labels


TIMING_PATTERNS = [
    re.compile(r"elapsed[:\s]+(\d+\.?\d*)\s*s", re.IGNORECASE),
    re.compile(r"time[:\s]+(\d+\.?\d*)\s*s", re.IGNORECASE),
    re.compile(r"(\d+\.?\d*)\s*s\s*(?:elapsed|total)", re.IGNORECASE),
    re.compile(r"(\d+\.?\d*)\s*seconds", re.IGNORECASE),
    re.compile(r"real\s+(\d+)m(\d+\.?\d*)s"),
]


def _parse_timing(log_path: str) -> Dict[str, float]:
    """Parse timing info from a log file.

    Returns dict mapping binary_name -> elapsed_seconds.
    """
    results: Dict[str, float] = {}
    current_binary = None

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            binary_match = re.search(r"(?:Processing|Analyzing|Binary)[:\s]+(\S+)", line, re.IGNORECASE)
            if binary_match:
                current_binary = binary_match.group(1)

            for pattern in TIMING_PATTERNS:
                m = pattern.search(line)
                if m:
                    if len(m.groups()) == 2:
                        minutes = float(m.group(1))
                        secs = float(m.group(2))
                        elapsed = minutes * 60 + secs
                    else:
                        elapsed = float(m.group(1))

                    if current_binary:
                        results[current_binary] = elapsed
                    break

    if not results:
        print(
            f"[ERROR] Could not parse any timing data from {log_path}. "
            "Supported formats: 'elapsed: X.Xs', 'time: Xs', 'Xs elapsed', "
            "'X seconds', 'real XmX.Xs'",
            file=sys.stderr,
        )
    return results


def measure_costs_manual(
    l1_cost: float, l2_cost: float, l3_cost: float
) -> Dict[str, Any]:
    return {
        "method": "manual",
        "costs": {"SKIP": 0, "L1": l1_cost, "L2": l2_cost, "L3": l3_cost},
        "ratio_description": f"L1:L2:L3 = {l1_cost}:{l2_cost}:{l3_cost}",
        "per_binary": None,
    }


def measure_costs_from_logs(
    ghidra_log: str,
    angr_log: str,
    labels_dir: str,
) -> Dict[str, Any]:
    import numpy as np

    entries = load_silver_labels(labels_dir)
    site_counts = {name: data["total"] for name, data in entries}

    ghidra_timings = _parse_timing(ghidra_log)
    angr_timings = _parse_timing(angr_log)

    per_binary: Dict[str, Dict[str, float]] = {}
    ghidra_per_site: List[float] = []
    angr_per_site: List[float] = []

    for binary_name, n_sites in site_counts.items():
        if n_sites == 0:
            continue
        entry: Dict[str, float] = {}
        if binary_name in ghidra_timings:
            gps = ghidra_timings[binary_name] / n_sites
            entry["ghidra_per_site_sec"] = round(gps, 4)
            ghidra_per_site.append(gps)
        if binary_name in angr_timings:
            aps = angr_timings[binary_name] / n_sites
            entry["angr_per_site_sec"] = round(aps, 4)
            angr_per_site.append(aps)
        if entry:
            per_binary[binary_name] = entry

    if ghidra_per_site and angr_per_site:
        l1_median = float(np.median(ghidra_per_site))
        l2_median = float(np.median(angr_per_site))
        ratio = round(l2_median / l1_median, 1) if l1_median > 0 else 5.0
        costs = {"SKIP": 0, "L1": 1.0, "L2": ratio, "L3": None}
    else:
        costs = {"SKIP": 0, "L1": 1.0, "L2": None, "L3": None}

    return {
        "method": "from_logs",
        "costs": costs,
        "per_binary": per_binary,
        "notes": "L3 pending Pin experiments. L1 normalized to 1.0.",
    }


def main():
    parser = argparse.ArgumentParser(description="Measure or specify cost ratios")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manual", action="store_true", help="Manual cost specification")
    mode.add_argument("--from-logs", action="store_true", help="Parse from timing logs")

    parser.add_argument("--l1-cost", type=float, default=1.0)
    parser.add_argument("--l2-cost", type=float, default=5.0)
    parser.add_argument("--l3-cost", type=float, default=20.0)

    parser.add_argument("--ghidra-log", type=str, default=None)
    parser.add_argument("--angr-log", type=str, default=None)
    parser.add_argument("--labels-dir", type=str, default="data/silver_labels")

    parser.add_argument("--output", default="data/calibration/cost_params.json")
    args = parser.parse_args()

    if args.manual:
        result = measure_costs_manual(args.l1_cost, args.l2_cost, args.l3_cost)
    else:
        if not args.ghidra_log or not args.angr_log:
            parser.error("--from-logs requires --ghidra-log and --angr-log")
        result = measure_costs_from_logs(
            args.ghidra_log, args.angr_log, args.labels_dir
        )

    print(json.dumps(result, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to: {args.output}")


if __name__ == "__main__":
    main()
