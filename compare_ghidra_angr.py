#!/usr/bin/env python3
"""Compare Ghidra vs angr jump table results for a single binary."""

import json
import sys

GHIDRA_FILE = "/home/peterzh/sec/playground/rl/extract_gt/ghidra_results_ssh_servers_gcc_O3/ssh.json"
ANGR_FILE = "/home/peterzh/sec/playground/rl/extract_gt/angr_results/ssh.json"


def load(path):
    with open(path) as f:
        data = json.load(f)
    jt_map = {}
    for jt in data["jump_tables"]:
        addr = int(jt["instruction_addr"], 16)
        targets = set(int(t, 16) for t in jt["targets"])
        jt_map[addr] = targets
    return jt_map


def main():
    ghidra = load(GHIDRA_FILE)
    angr_ = load(ANGR_FILE)

    ghidra_addrs = set(ghidra.keys())
    angr_addrs = set(angr_.keys())

    shared = sorted(ghidra_addrs & angr_addrs)
    ghidra_only = sorted(ghidra_addrs - angr_addrs)
    angr_only = sorted(angr_addrs - ghidra_addrs)

    print("=" * 70)
    print(f"Ghidra: {len(ghidra)} jump tables    angr: {len(angr_)} jump tables")
    print(f"Shared addresses: {len(shared)}")
    print(f"Ghidra-only:      {len(ghidra_only)}")
    print(f"angr-only:        {len(angr_only)}")
    print("=" * 70)

    # --- Shared: compare targets ---
    print(f"\n{'='*70}")
    print("SHARED JUMP TABLES — target comparison")
    print(f"{'='*70}")

    total_g, total_a, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0

    for addr in shared:
        g_targets = ghidra[addr]
        a_targets = angr_[addr]
        tp = g_targets & a_targets
        fp_g = g_targets - a_targets  # in Ghidra but not angr
        fp_a = a_targets - g_targets  # in angr but not Ghidra

        total_g += len(g_targets)
        total_a += len(a_targets)
        total_tp += len(tp)
        total_fp += len(fp_g)
        total_fn += len(fp_a)

        status = "EXACT MATCH" if not fp_g and not fp_a else "DIFFER"
        print(f"\n  0x{addr:08x}  [{status}]")
        print(f"    Ghidra targets: {len(g_targets)}   angr targets: {len(a_targets)}   overlap: {len(tp)}")
        if fp_g:
            print(f"    Ghidra extra ({len(fp_g)}): {sorted(f'0x{t:08x}' for t in fp_g)}")
        if fp_a:
            print(f"    angr extra   ({len(fp_a)}): {sorted(f'0x{t:08x}' for t in fp_a)}")

    print(f"\n  --- Shared totals ---")
    print(f"  Ghidra targets: {total_g}   angr targets: {total_a}")
    print(f"  Overlap: {total_tp}   Ghidra-extra: {total_fp}   angr-extra: {total_fn}")

    # --- Ghidra-only ---
    if ghidra_only:
        print(f"\n{'='*70}")
        print(f"GHIDRA-ONLY jump tables ({len(ghidra_only)})")
        print(f"{'='*70}")
        for addr in ghidra_only:
            print(f"  0x{addr:08x}  targets={len(ghidra[addr])}")

    # --- angr-only ---
    if angr_only:
        print(f"\n{'='*70}")
        print(f"ANGR-ONLY jump tables ({len(angr_only)})")
        print(f"{'='*70}")
        for addr in angr_only:
            print(f"  0x{addr:08x}  targets={len(angr_[addr])}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Ghidra total JTs: {len(ghidra)}")
    print(f"  angr   total JTs: {len(angr_)}")
    print(f"  Shared:           {len(shared)} / {len(ghidra_addrs | angr_addrs)}  "
          f"(Jaccard = {len(shared)/max(1,len(ghidra_addrs|angr_addrs)):.1%})")

    if shared:
        exact = sum(1 for a in shared if ghidra[a] == angr_[a])
        print(f"  Exact target match: {exact} / {len(shared)}")

    all_ghidra_targets = sum(len(v) for v in ghidra.values())
    all_angr_targets = sum(len(v) for v in angr_.values())
    print(f"  Total targets — Ghidra: {all_ghidra_targets}   angr: {all_angr_targets}")


if __name__ == "__main__":
    main()
