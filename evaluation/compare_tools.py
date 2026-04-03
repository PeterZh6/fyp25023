#!/usr/bin/env python3
"""
Compare Ghidra vs angr indirect flow results for one or more binaries.

Usage:
    python -m evaluation.compare_tools ssh
    python -m evaluation.compare_tools ssh scp openssl
    python -m evaluation.compare_tools --all
    python -m evaluation.compare_tools --all --ghidra-dir data/ghidra_results --angr-dir data/angr_results
"""

import json
import os
import re
import sys
import argparse

from calibration.config import filter_binary_names


def load_indirect_flows(path):
    """Load indirect flows JSON, return dict {address: (type, set(targets))}."""
    with open(path) as f:
        data = json.load(f)

    flow_map = {}
    for flow in data.get("indirect_flows", []):
        addr = int(flow["address"], 16)
        targets = set(int(t, 16) for t in flow["targets"])
        flow_map[addr] = (flow["type"], targets)
    return flow_map


def discover_binaries(ghidra_dir, angr_dir):
    """Return sorted list of binary names that have BOTH ghidra and angr results."""
    ghidra_names = set()
    angr_names = set()

    for fname in os.listdir(ghidra_dir):
        m = re.match(r"^(.+)_ghidra\.json$", fname)
        if m:
            ghidra_names.add(m.group(1))

    for fname in os.listdir(angr_dir):
        m = re.match(r"^(.+)_angr\.json$", fname)
        if m:
            angr_names.add(m.group(1))

    common = ghidra_names & angr_names
    only_ghidra = ghidra_names - angr_names
    only_angr = angr_names - ghidra_names

    if only_ghidra:
        print(f"[info] Ghidra-only (no angr match): {sorted(only_ghidra)}")
    if only_angr:
        print(f"[info] angr-only (no Ghidra match): {sorted(only_angr)}")

    return filter_binary_names(sorted(common))


def compare_one(binary, ghidra_dir, angr_dir, verbose=True):
    """
    Compare Ghidra vs angr for a single binary.
    Returns a metrics dict, or None if files are missing.
    """
    ghidra_path = os.path.join(ghidra_dir, f"{binary}_ghidra.json")
    angr_path = os.path.join(angr_dir, f"{binary}_angr.json")

    if not os.path.exists(ghidra_path):
        print(f"[skip] Ghidra result not found: {ghidra_path}")
        return None
    if not os.path.exists(angr_path):
        print(f"[skip] angr result not found: {angr_path}")
        return None

    ghidra = load_indirect_flows(ghidra_path)
    angr_ = load_indirect_flows(angr_path)

    ghidra_addrs = set(ghidra.keys())
    angr_addrs = set(angr_.keys())

    shared = sorted(ghidra_addrs & angr_addrs)
    ghidra_only = sorted(ghidra_addrs - angr_addrs)
    angr_only = sorted(angr_addrs - ghidra_addrs)

    if verbose:
        print("=" * 70)
        print(f"Binary: {binary}")
        print(f"Ghidra: {len(ghidra)} indirect flows    angr: {len(angr_)} indirect flows")
        print(f"Shared addresses: {len(shared)}")
        print(f"Ghidra-only:      {len(ghidra_only)}")
        print(f"angr-only:        {len(angr_only)}")
        print("=" * 70)

    total_g, total_a, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0

    if verbose and shared:
        print(f"\n{'='*70}")
        print("SHARED INDIRECT FLOWS — target comparison")
        print(f"{'='*70}")

    for addr in shared:
        g_type, g_targets = ghidra[addr]
        a_type, a_targets = angr_[addr]
        tp = g_targets & a_targets
        fp_g = g_targets - a_targets
        fp_a = a_targets - g_targets

        total_g += len(g_targets)
        total_a += len(a_targets)
        total_tp += len(tp)
        total_fp += len(fp_g)
        total_fn += len(fp_a)

        if verbose:
            status = "EXACT MATCH" if not fp_g and not fp_a else "DIFFER"
            type_str = f"ghidra={g_type} angr={a_type}"
            print(f"\n  0x{addr:08x}  [{status}]  ({type_str})")
            print(f"    Ghidra targets: {len(g_targets)}   angr targets: {len(a_targets)}   overlap: {len(tp)}")
            if fp_g:
                print(f"    Ghidra extra ({len(fp_g)}): {sorted(f'0x{t:08x}' for t in fp_g)}")
            if fp_a:
                print(f"    angr extra   ({len(fp_a)}): {sorted(f'0x{t:08x}' for t in fp_a)}")

    if verbose:
        print(f"\n  --- Shared totals ---")
        print(f"  Ghidra targets: {total_g}   angr targets: {total_a}")
        print(f"  Overlap: {total_tp}   Ghidra-extra: {total_fp}   angr-extra: {total_fn}")

    if verbose and ghidra_only:
        print(f"\n{'='*70}")
        print(f"GHIDRA-ONLY flows ({len(ghidra_only)})")
        print(f"{'='*70}")
        for addr in ghidra_only:
            ft, tgts = ghidra[addr]
            print(f"  0x{addr:08x}  type={ft}  targets={len(tgts)}")

    if verbose and angr_only:
        print(f"\n{'='*70}")
        print(f"ANGR-ONLY flows ({len(angr_only)})")
        print(f"{'='*70}")
        for addr in angr_only:
            ft, tgts = angr_[addr]
            print(f"  0x{addr:08x}  type={ft}  targets={len(tgts)}")

    all_addrs = ghidra_addrs | angr_addrs
    jaccard = len(shared) / max(1, len(all_addrs))
    exact = sum(1 for a in shared if ghidra[a][1] == angr_[a][1]) if shared else 0

    all_g_targets = sum(len(v[1]) for v in ghidra.values())
    all_a_targets = sum(len(v[1]) for v in angr_.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None

    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Ghidra total: {len(ghidra)}")
        print(f"  angr   total: {len(angr_)}")
        print(f"  Shared:       {len(shared)} / {len(all_addrs)}  (Jaccard = {jaccard:.1%})")
        if shared:
            print(f"  Exact target match: {exact} / {len(shared)}")
        print(f"  Total targets — Ghidra: {all_g_targets}   angr: {all_a_targets}")
        if precision is not None:
            print(f"  Shared precision: {precision:.1%}")
        if recall is not None:
            print(f"  Shared recall:    {recall:.1%}")

    return {
        "binary": binary,
        "ghidra_flows": len(ghidra),
        "angr_flows": len(angr_),
        "shared": len(shared),
        "ghidra_only": len(ghidra_only),
        "angr_only": len(angr_only),
        "union": len(all_addrs),
        "jaccard": jaccard,
        "exact_match": exact,
        "ghidra_targets": all_g_targets,
        "angr_targets": all_a_targets,
        "overlap_targets": total_tp,
        "precision": precision,
        "recall": recall,
    }


def print_summary_table(results):
    """Print a compact summary table across all binaries."""
    print("\n")
    print("=" * 100)
    print("BATCH SUMMARY")
    print("=" * 100)

    header = (
        f"{'Binary':<15} {'Ghidra':>7} {'angr':>7} {'Shared':>7} "
        f"{'Union':>7} {'Jaccard':>8} {'Exact':>7} "
        f"{'Prec':>7} {'Recall':>7}"
    )
    print(header)
    print("-" * 100)

    totals = dict(
        ghidra_flows=0, angr_flows=0, shared=0, union=0,
        exact_match=0, overlap_targets=0,
        ghidra_extra=0, angr_extra=0,
    )

    for r in results:
        prec_str = f"{r['precision']:.1%}" if r["precision"] is not None else "N/A"
        rec_str = f"{r['recall']:.1%}" if r["recall"] is not None else "N/A"
        print(
            f"{r['binary']:<15} {r['ghidra_flows']:>7} {r['angr_flows']:>7} "
            f"{r['shared']:>7} {r['union']:>7} {r['jaccard']:>7.1%} "
            f"{r['exact_match']:>7} {prec_str:>7} {rec_str:>7}"
        )
        totals["ghidra_flows"] += r["ghidra_flows"]
        totals["angr_flows"] += r["angr_flows"]
        totals["shared"] += r["shared"]
        totals["union"] += r["union"]
        totals["exact_match"] += r["exact_match"]
        totals["overlap_targets"] += r["overlap_targets"]
        if r["precision"] is not None:
            totals["ghidra_extra"] += round(r["overlap_targets"] / r["precision"]) - r["overlap_targets"] if r["precision"] > 0 else 0
        if r["recall"] is not None:
            totals["angr_extra"] += round(r["overlap_targets"] / r["recall"]) - r["overlap_targets"] if r["recall"] > 0 else 0

    print("-" * 100)

    agg_jaccard = totals["shared"] / max(1, totals["union"])
    tp = totals["overlap_targets"]
    agg_prec = tp / (tp + totals["ghidra_extra"]) if (tp + totals["ghidra_extra"]) > 0 else None
    agg_rec = tp / (tp + totals["angr_extra"]) if (tp + totals["angr_extra"]) > 0 else None
    prec_str = f"{agg_prec:.1%}" if agg_prec is not None else "N/A"
    rec_str = f"{agg_rec:.1%}" if agg_rec is not None else "N/A"

    print(
        f"{'TOTAL':<15} {totals['ghidra_flows']:>7} {totals['angr_flows']:>7} "
        f"{totals['shared']:>7} {totals['union']:>7} {agg_jaccard:>7.1%} "
        f"{totals['exact_match']:>7} {prec_str:>7} {rec_str:>7}"
    )
    print("=" * 100)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Compare Ghidra vs angr indirect flows")
    parser.add_argument("binaries", nargs="*", help="Binary names (e.g. ssh scp openssl)")
    parser.add_argument("--all", action="store_true",
                        help="Auto-discover and compare all binaries with both results")
    parser.add_argument("--ghidra-dir",
                        default=os.path.join(project_root, "data", "ghidra_results"),
                        help="Directory containing Ghidra JSON results")
    parser.add_argument("--angr-dir",
                        default=os.path.join(project_root, "data", "angr_results"),
                        help="Directory containing angr JSON results")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress per-binary details, only show summary table")
    args = parser.parse_args()

    if args.all:
        binaries = discover_binaries(args.ghidra_dir, args.angr_dir)
        if not binaries:
            print("No matching binaries found in both directories.")
            sys.exit(1)
        print(f"[info] Discovered {len(binaries)} binaries: {binaries}\n")
    elif args.binaries:
        binaries = filter_binary_names(args.binaries)
    else:
        parser.error("Provide binary names or use --all")

    batch = len(binaries) > 1
    verbose = not args.quiet

    results = []
    for binary in binaries:
        if batch and verbose:
            print(f"\n{'#'*70}")
            print(f"# {binary}")
            print(f"{'#'*70}")
        r = compare_one(binary, args.ghidra_dir, args.angr_dir, verbose=verbose)
        if r is not None:
            results.append(r)

    if not results:
        print("No valid comparisons.")
        sys.exit(1)

    if batch or args.quiet:
        print_summary_table(results)


if __name__ == "__main__":
    main()
