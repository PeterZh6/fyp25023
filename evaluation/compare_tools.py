#!/usr/bin/env python3
"""
Compare Ghidra vs angr indirect flow results for a single binary.

Usage:
    python -m evaluation.compare_tools <binary_name>
    python -m evaluation.compare_tools ssh
    python -m evaluation.compare_tools --ghidra-dir data/ghidra_results --angr-dir data/angr_results ssh
"""

import json
import os
import sys
import argparse


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


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Compare Ghidra vs angr indirect flows")
    parser.add_argument("binary", help="Binary name (e.g. ssh, scp)")
    parser.add_argument("--ghidra-dir",
                        default=os.path.join(project_root, "data", "ghidra_results"),
                        help="Directory containing Ghidra JSON results")
    parser.add_argument("--angr-dir",
                        default=os.path.join(project_root, "data", "angr_results"),
                        help="Directory containing angr JSON results")
    args = parser.parse_args()

    ghidra_path = os.path.join(args.ghidra_dir, f"{args.binary}_ghidra.json")
    angr_path = os.path.join(args.angr_dir, f"{args.binary}_angr.json")

    if not os.path.exists(ghidra_path):
        print(f"Ghidra result not found: {ghidra_path}")
        sys.exit(1)
    if not os.path.exists(angr_path):
        print(f"angr result not found: {angr_path}")
        sys.exit(1)

    ghidra = load_indirect_flows(ghidra_path)
    angr_ = load_indirect_flows(angr_path)

    ghidra_addrs = set(ghidra.keys())
    angr_addrs = set(angr_.keys())

    shared = sorted(ghidra_addrs & angr_addrs)
    ghidra_only = sorted(ghidra_addrs - angr_addrs)
    angr_only = sorted(angr_addrs - ghidra_addrs)

    print("=" * 70)
    print(f"Binary: {args.binary}")
    print(f"Ghidra: {len(ghidra)} indirect flows    angr: {len(angr_)} indirect flows")
    print(f"Shared addresses: {len(shared)}")
    print(f"Ghidra-only:      {len(ghidra_only)}")
    print(f"angr-only:        {len(angr_only)}")
    print("=" * 70)

    # Shared: compare targets
    print(f"\n{'='*70}")
    print("SHARED INDIRECT FLOWS — target comparison")
    print(f"{'='*70}")

    total_g, total_a, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0

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

        status = "EXACT MATCH" if not fp_g and not fp_a else "DIFFER"
        type_str = f"ghidra={g_type} angr={a_type}"
        print(f"\n  0x{addr:08x}  [{status}]  ({type_str})")
        print(f"    Ghidra targets: {len(g_targets)}   angr targets: {len(a_targets)}   overlap: {len(tp)}")
        if fp_g:
            print(f"    Ghidra extra ({len(fp_g)}): {sorted(f'0x{t:08x}' for t in fp_g)}")
        if fp_a:
            print(f"    angr extra   ({len(fp_a)}): {sorted(f'0x{t:08x}' for t in fp_a)}")

    print(f"\n  --- Shared totals ---")
    print(f"  Ghidra targets: {total_g}   angr targets: {total_a}")
    print(f"  Overlap: {total_tp}   Ghidra-extra: {total_fp}   angr-extra: {total_fn}")

    if ghidra_only:
        print(f"\n{'='*70}")
        print(f"GHIDRA-ONLY flows ({len(ghidra_only)})")
        print(f"{'='*70}")
        for addr in ghidra_only:
            ft, tgts = ghidra[addr]
            print(f"  0x{addr:08x}  type={ft}  targets={len(tgts)}")

    if angr_only:
        print(f"\n{'='*70}")
        print(f"ANGR-ONLY flows ({len(angr_only)})")
        print(f"{'='*70}")
        for addr in angr_only:
            ft, tgts = angr_[addr]
            print(f"  0x{addr:08x}  type={ft}  targets={len(tgts)}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    all_addrs = ghidra_addrs | angr_addrs
    jaccard = len(shared) / max(1, len(all_addrs))
    print(f"  Ghidra total: {len(ghidra)}")
    print(f"  angr   total: {len(angr_)}")
    print(f"  Shared:       {len(shared)} / {len(all_addrs)}  (Jaccard = {jaccard:.1%})")

    if shared:
        exact = sum(1 for a in shared if ghidra[a][1] == angr_[a][1])
        print(f"  Exact target match: {exact} / {len(shared)}")

    all_g_targets = sum(len(v[1]) for v in ghidra.values())
    all_a_targets = sum(len(v[1]) for v in angr_.values())
    print(f"  Total targets — Ghidra: {all_g_targets}   angr: {all_a_targets}")

    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
        print(f"  Shared precision: {precision:.1%}")
    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
        print(f"  Shared recall:    {recall:.1%}")


if __name__ == "__main__":
    main()
