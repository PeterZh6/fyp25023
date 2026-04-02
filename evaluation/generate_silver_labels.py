#!/usr/bin/env python3
"""Generate silver difficulty labels for indirect flow sites by comparing Ghidra and angr results.

For each site, labels are assigned based on cross-tool agreement:
  easy            - both tools resolved targets with overlap
  medium_conflict - both resolved but zero overlap
  medium_partial  - only one tool resolved targets
  hard            - neither tool resolved targets

Final three-class mapping: easy, medium (conflict + partial), hard.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    by_addr = {}
    for entry in data.get("indirect_flows", []):
        by_addr[entry["address"]] = entry
    return data, by_addr


def label_site(ghidra_entry, angr_entry):
    g_targets = set(ghidra_entry["targets"]) if ghidra_entry else set()
    a_targets = set(angr_entry["targets"]) if angr_entry else set()
    g_resolved = len(g_targets) > 0
    a_resolved = len(a_targets) > 0

    if g_resolved and a_resolved:
        if g_targets & a_targets:
            return "easy"
        else:
            return "medium_conflict"
    elif g_resolved or a_resolved:
        return "medium_partial"
    else:
        return "hard"


DETAIL_TO_MERGED = {
    "easy": "easy",
    "medium_conflict": "medium",
    "medium_partial": "medium",
    "hard": "hard",
}


def overlap_ratio(ghidra_entry, angr_entry):
    g = set(ghidra_entry["targets"]) if ghidra_entry else set()
    a = set(angr_entry["targets"]) if angr_entry else set()
    union = g | a
    if not union:
        return None
    return len(g & a) / len(union)


def resolve_type(ghidra_entry, angr_entry):
    g_type = ghidra_entry.get("type") if ghidra_entry else None
    a_type = angr_entry.get("type") if angr_entry else None

    if g_type:
        chosen, source = g_type, "ghidra"
    elif a_type:
        chosen, source = a_type, "angr"
    else:
        chosen, source = "unknown", "none"

    return chosen, source, g_type, a_type


def resolve_function(ghidra_entry, angr_entry):
    """Pick function name — prefer Ghidra, fall back to angr."""
    if ghidra_entry and ghidra_entry.get("function"):
        return ghidra_entry["function"]
    if angr_entry and angr_entry.get("function"):
        return angr_entry["function"]
    return "unknown"


def process_binary(ghidra_path, angr_path, binary_name):
    _, ghidra_map = load_results(ghidra_path)
    _, angr_map = load_results(angr_path)

    all_addrs = sorted(set(ghidra_map.keys()) | set(angr_map.keys()))

    sites = []
    for addr in all_addrs:
        g_entry = ghidra_map.get(addr)
        a_entry = angr_map.get(addr)

        detail = label_site(g_entry, a_entry)
        merged = DETAIL_TO_MERGED[detail]

        chosen_type, type_source, g_type, a_type = resolve_type(g_entry, a_entry)
        func = resolve_function(g_entry, a_entry)

        ratio = overlap_ratio(g_entry, a_entry) if detail in ("easy", "medium_conflict") else None

        g_targets = sorted(g_entry["targets"]) if g_entry else []
        a_targets = sorted(a_entry["targets"]) if a_entry else []

        sites.append({
            "address": addr,
            "type": chosen_type,
            "type_source": type_source,
            "ghidra_type": g_type,
            "angr_type": a_type,
            "type_agree": g_type == a_type if (g_type and a_type) else None,
            "function": func,
            "label": merged,
            "label_detail": detail,
            "ghidra_targets": g_targets,
            "angr_targets": a_targets,
            "overlap_ratio": round(ratio, 4) if ratio is not None else None,
            "ghidra_found": g_entry is not None,
            "angr_found": a_entry is not None,
        })

    summary = {"easy": 0, "medium": 0, "medium_conflict": 0, "medium_partial": 0, "hard": 0}
    for s in sites:
        summary[s["label"]] += 1
        if s["label_detail"] in ("medium_conflict", "medium_partial"):
            summary[s["label_detail"]] += 1

    result = {
        "binary": binary_name,
        "total": len(sites),
        "summary": summary,
        "sites": sites,
    }
    return result


def discover_binaries(ghidra_dir, angr_dir):
    """Find binaries that have results in both directories."""
    ghidra_names = {}
    for f in os.listdir(ghidra_dir):
        if f.endswith("_ghidra.json"):
            name = f[: -len("_ghidra.json")]
            ghidra_names[name] = os.path.join(ghidra_dir, f)

    angr_names = {}
    for f in os.listdir(angr_dir):
        if f.endswith("_angr.json"):
            name = f[: -len("_angr.json")]
            angr_names[name] = os.path.join(angr_dir, f)

    common = sorted(set(ghidra_names.keys()) & set(angr_names.keys()))
    return [(name, ghidra_names[name], angr_names[name]) for name in common]


def write_summary_csv(all_results, output_dir):
    csv_path = os.path.join(output_dir, "silver_labels_summary.csv")
    fieldnames = [
        "binary", "total", "easy", "medium", "medium_conflict", "medium_partial", "hard",
        "easy_pct", "medium_pct", "hard_pct",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            s = r["summary"]
            total = r["total"] or 1
            writer.writerow({
                "binary": r["binary"],
                "total": r["total"],
                "easy": s["easy"],
                "medium": s["medium"],
                "medium_conflict": s["medium_conflict"],
                "medium_partial": s["medium_partial"],
                "hard": s["hard"],
                "easy_pct": round(100 * s["easy"] / total, 1),
                "medium_pct": round(100 * s["medium"] / total, 1),
                "hard_pct": round(100 * s["hard"] / total, 1),
            })
    return csv_path


def print_summary_table(all_results):
    header = f"{'Binary':<45} {'Total':>6} {'Easy':>6} {'Med':>6} {'(conf)':>6} {'(part)':>6} {'Hard':>6}  {'E%':>5} {'M%':>5} {'H%':>5}"
    print("\n" + "=" * len(header))
    print("Silver Label Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    totals = {"total": 0, "easy": 0, "medium": 0, "mc": 0, "mp": 0, "hard": 0}
    for r in all_results:
        s = r["summary"]
        t = r["total"] or 1
        print(
            f"{r['binary']:<45} {r['total']:>6} {s['easy']:>6} {s['medium']:>6} "
            f"{s['medium_conflict']:>6} {s['medium_partial']:>6} {s['hard']:>6}  "
            f"{100*s['easy']/t:>5.1f} {100*s['medium']/t:>5.1f} {100*s['hard']/t:>5.1f}"
        )
        totals["total"] += r["total"]
        totals["easy"] += s["easy"]
        totals["medium"] += s["medium"]
        totals["mc"] += s["medium_conflict"]
        totals["mp"] += s["medium_partial"]
        totals["hard"] += s["hard"]

    print("-" * len(header))
    gt = totals["total"] or 1
    print(
        f"{'TOTAL':<45} {totals['total']:>6} {totals['easy']:>6} {totals['medium']:>6} "
        f"{totals['mc']:>6} {totals['mp']:>6} {totals['hard']:>6}  "
        f"{100*totals['easy']/gt:>5.1f} {100*totals['medium']/gt:>5.1f} {100*totals['hard']/gt:>5.1f}"
    )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="Generate silver difficulty labels by comparing Ghidra and angr results."
    )
    parser.add_argument("--ghidra-dir", required=True, help="Directory containing *_ghidra.json files")
    parser.add_argument("--angr-dir", required=True, help="Directory containing *_angr.json files")
    parser.add_argument("--output-dir", required=True, help="Output directory for label JSONs and summary CSV")
    args = parser.parse_args()

    binaries = discover_binaries(args.ghidra_dir, args.angr_dir)
    if not binaries:
        print("No matching binaries found in both directories.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(binaries)} binaries with results from both tools.")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for name, ghidra_path, angr_path in binaries:
        result = process_binary(ghidra_path, angr_path, name)
        out_path = os.path.join(args.output_dir, f"{name}_labels.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  {name}: {result['total']} sites → {out_path}")
        all_results.append(result)

    csv_path = write_summary_csv(all_results, args.output_dir)
    print(f"\nSummary CSV → {csv_path}")

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
