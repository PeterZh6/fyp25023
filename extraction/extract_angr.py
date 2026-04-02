#!/usr/bin/env python3
"""
Extract indirect control-flow information from ARM32 ELF binaries using angr's CFGFast.
Output format aligned with Ghidra ExportIndirectFlows JSON for evaluation.
"""

import angr
import json
import os
import sys
import time
import logging
import argparse

logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)
logging.getLogger("pyvex").setLevel(logging.CRITICAL)

GHIDRA_ARM_PIE_BASE = 0x10000
TIMEOUT = 300


def clear_thumb_bit(addr):
    return addr & ~1


def format_addr(addr):
    return f"{clear_thumb_bit(addr):08x}"


def extract_indirect_flows(binary_path, binary_name, timeout=300):
    """
    Load a binary with angr, run CFGFast, and extract all indirect flows
    (both jumps and calls).
    """
    print(f"  Loading {binary_path} ...")

    proj_probe = angr.Project(binary_path, auto_load_libs=False)
    is_pie = proj_probe.loader.main_object.pic
    linked_base = proj_probe.loader.main_object.linked_base

    if is_pie:
        proj = angr.Project(binary_path, auto_load_libs=False,
                            main_opts={"base_addr": GHIDRA_ARM_PIE_BASE})
        print(f"  PIE binary, rebased to 0x{GHIDRA_ARM_PIE_BASE:x} (Ghidra default)")
    else:
        proj = proj_probe
        print(f"  Non-PIE binary, linked base 0x{linked_base:x}")

    print(f"  Running CFGFast (timeout={timeout}s) ...")
    start = time.time()

    cfg = proj.analyses.CFGFast(
        normalize=True,
        resolve_indirect_jumps=True,
    )

    elapsed = time.time() - start
    print(f"  CFGFast completed in {elapsed:.1f}s")

    indirect_flows = []
    all_ij = cfg.indirect_jumps

    for addr, ij in all_ij.items():
        inst_addr = clear_thumb_bit(ij.ins_addr)
        targets_raw = ij.resolved_targets or []
        targets = sorted(set(clear_thumb_bit(t) for t in targets_raw))

        jumpkind = getattr(ij, "jumpkind", "Ijk_Boring")
        if "Call" in jumpkind:
            flow_type = "call"
        else:
            flow_type = "jump"

        indirect_flows.append({
            "address": format_addr(inst_addr),
            "type": flow_type,
            "function": "",
            "targets": [format_addr(t) for t in targets],
        })

    indirect_flows.sort(key=lambda f: f["address"])

    arch_name = proj.arch.name
    result = {
        "binary": binary_name,
        "arch": arch_name,
        "indirect_flows": indirect_flows,
    }

    jumps = sum(1 for f in indirect_flows if f["type"] == "jump")
    calls = sum(1 for f in indirect_flows if f["type"] == "call")
    resolved = sum(1 for f in indirect_flows if f["targets"])
    print(f"  Found {len(indirect_flows)} indirect flows "
          f"({jumps} jumps, {calls} calls, {resolved} resolved)")

    return result


def find_binaries(directory):
    """Find all ELF files in a directory."""
    binaries = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                if f.read(4) == b"\x7fELF":
                    binaries.append((name, path))
        except (PermissionError, IsADirectoryError):
            pass
    return binaries


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Extract indirect flows via angr")
    parser.add_argument("-d", "--binary-dir",
                        default=os.path.join(project_root, "data", "binaries", "ssh_servers_gcc_O3"),
                        help="Directory containing ELF binaries")
    parser.add_argument("-o", "--output-dir",
                        default=os.path.join(project_root, "data", "angr_results"),
                        help="Output directory for JSON results")
    parser.add_argument("-t", "--timeout", type=int, default=TIMEOUT,
                        help="Timeout per binary in seconds")
    parser.add_argument("binaries", nargs="*",
                        help="Specific binary names to analyze (default: all in directory)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_binaries = find_binaries(args.binary_dir)
    if args.binaries:
        name_set = set(args.binaries)
        all_binaries = [(n, p) for n, p in all_binaries if n in name_set]

    print(f"Binary dir:  {args.binary_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Binaries:    {len(all_binaries)}")
    print()

    all_results = []

    for name, path in all_binaries:
        print(f"[{name}]")
        try:
            result = extract_indirect_flows(path, name, timeout=args.timeout)
            out_path = os.path.join(args.output_dir, f"{name}_angr.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved to {out_path}")
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 50)
    print("Summary:")
    print("=" * 50)
    total = sum(len(r["indirect_flows"]) for r in all_results)
    print(f"  Binaries analyzed: {len(all_results)}/{len(all_binaries)}")
    print(f"  Total indirect flows: {total}")
    for r in all_results:
        n = len(r["indirect_flows"])
        print(f"    {r['binary']}: {n} flows")


if __name__ == "__main__":
    main()
