"""Compute difficulty distributions from silver label data.

Reads all silver label JSON files and computes per-binary and mixed
difficulty distributions (easy / medium / hard).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

from calibration.config import filter_binary_entries


def load_silver_labels(input_dir: str) -> list[tuple[str, dict]]:
    """Load all silver label JSON files from directory.

    Returns list of (binary_name, data_dict) tuples.
    """
    results = []
    label_dir = Path(input_dir)
    for fpath in sorted(label_dir.glob("*_labels.json")):
        with open(fpath) as f:
            data = json.load(f)
        binary_name = data["binary"]
        results.append((binary_name, data))
    return filter_binary_entries(results, lambda entry: entry[0])


def compute_distributions(input_dir: str) -> Dict[str, Any]:
    """Compute difficulty distributions for all binaries."""
    entries = load_silver_labels(input_dir)

    per_binary: Dict[str, Any] = {}
    total_easy = 0
    total_medium = 0
    total_hard = 0

    for binary_name, data in entries:
        summary = data["summary"]
        easy = summary["easy"]
        hard = summary["hard"]
        medium = summary["medium"]

        sites = data["sites"]
        medium_from_sites = sum(1 for s in sites if s["label"] == "medium")
        if medium_from_sites != medium:
            print(
                f"[WARNING] {binary_name}: summary medium={medium} "
                f"but sites count={medium_from_sites}",
                file=sys.stderr,
            )

        total = easy + medium + hard
        if total != data["total"]:
            print(
                f"[WARNING] {binary_name}: easy+medium+hard={total} "
                f"!= data.total={data['total']}",
                file=sys.stderr,
            )

        per_binary[binary_name] = {
            "total_sites": total,
            "easy": round(easy / total, 4) if total > 0 else 0,
            "medium": round(medium / total, 4) if total > 0 else 0,
            "hard": round(hard / total, 4) if total > 0 else 0,
            "counts": {"easy": easy, "medium": medium, "hard": hard},
        }

        total_easy += easy
        total_medium += medium
        total_hard += hard

    grand_total = total_easy + total_medium + total_hard

    binary_weights = {
        name: round(info["total_sites"] / grand_total, 4)
        for name, info in per_binary.items()
    }
    mixed_all = {
        "total_sites": grand_total,
        "easy": round(total_easy / grand_total, 4),
        "medium": round(total_medium / grand_total, 4),
        "hard": round(total_hard / grand_total, 4),
        "binary_weights": binary_weights,
    }

    cpp_keywords = ["xalan", "dealii"]
    no_cpp_easy = 0
    no_cpp_medium = 0
    no_cpp_hard = 0
    excluded = []
    for name, info in per_binary.items():
        if any(kw in name.lower() for kw in cpp_keywords):
            excluded.append(name)
            continue
        no_cpp_easy += info["counts"]["easy"]
        no_cpp_medium += info["counts"]["medium"]
        no_cpp_hard += info["counts"]["hard"]

    no_cpp_total = no_cpp_easy + no_cpp_medium + no_cpp_hard
    mixed_no_cpp = {
        "total_sites": no_cpp_total,
        "easy": round(no_cpp_easy / no_cpp_total, 4) if no_cpp_total > 0 else 0,
        "medium": round(no_cpp_medium / no_cpp_total, 4) if no_cpp_total > 0 else 0,
        "hard": round(no_cpp_hard / no_cpp_total, 4) if no_cpp_total > 0 else 0,
        "excluded": excluded,
        "note": "Excludes C++ vtable-dominated binaries",
    }

    # Spot-check gcc
    if "gcc_base.arm32-gcc81-O3" in per_binary:
        gcc = per_binary["gcc_base.arm32-gcc81-O3"]
        checks = [
            ("easy", 0.512, gcc["easy"]),
            ("medium", 0.213, gcc["medium"]),
            ("hard", 0.275, gcc["hard"]),
        ]
        for label, expected, actual in checks:
            if abs(actual - expected) > 0.005:
                print(
                    f"[WARNING] gcc spot-check: {label} expected ~{expected}, "
                    f"got {actual}",
                    file=sys.stderr,
                )

    return {
        "per_binary": per_binary,
        "mixed_all": mixed_all,
        "mixed_no_cpp": mixed_no_cpp,
    }


def print_summary_table(result: Dict[str, Any]) -> None:
    """Print human-readable summary sorted by total descending."""
    per_binary = result["per_binary"]
    rows = sorted(per_binary.items(), key=lambda x: x[1]["total_sites"], reverse=True)

    print(f"\n{'Binary':<45s} {'Total':>6s} {'Easy%':>7s} {'Med%':>7s} {'Hard%':>7s}")
    print("-" * 75)
    for name, info in rows:
        print(
            f"{name:<45s} {info['total_sites']:>6d} "
            f"{info['easy']*100:>6.1f}% {info['medium']*100:>6.1f}% "
            f"{info['hard']*100:>6.1f}%"
        )

    mixed = result["mixed_all"]
    print("-" * 75)
    print(
        f"{'MIXED (all)':<45s} {mixed['total_sites']:>6d} "
        f"{mixed['easy']*100:>6.1f}% {mixed['medium']*100:>6.1f}% "
        f"{mixed['hard']*100:>6.1f}%"
    )
    no_cpp = result["mixed_no_cpp"]
    print(
        f"{'MIXED (no C++)':<45s} {no_cpp['total_sites']:>6d} "
        f"{no_cpp['easy']*100:>6.1f}% {no_cpp['medium']*100:>6.1f}% "
        f"{no_cpp['hard']*100:>6.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute difficulty distributions from silver labels"
    )
    parser.add_argument(
        "--input-dir", default="data/silver_labels",
        help="Directory containing *_labels.json files",
    )
    parser.add_argument(
        "--output", default="data/calibration/difficulty_distributions.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    result = compute_distributions(args.input_dir)
    print_summary_table(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to: {args.output}")


if __name__ == "__main__":
    main()
