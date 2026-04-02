#!/usr/bin/env python3
"""
Extract jump table information from ARM32 ELF binaries using angr's CFGFast.
Output format aligned with Ghidra ground truth JSON for evaluation.
"""

import angr
import json
import os
import sys
import time
import logging

# Suppress angr's verbose logging (it's VERY noisy)
logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)
logging.getLogger("pyvex").setLevel(logging.CRITICAL)

# ============================================================
# Configuration - edit these paths to match your setup
# ============================================================

# Directory containing binary subdirectories
BINARY_BASE_DIR = "/home/peterzh/sec/playground/rl/extract_gt/gts/ssh_servers_gcc_O3/"

# Output directory for JSON results
OUTPUT_DIR = "/home/peterzh/sec/playground/rl/extract_gt/angr_results/"

# List of target binaries to analyze
TARGET_BINARIES = [
    "scp",
    "sftp",
    "lighttpd",
    "ssh",
    "ssh-add",
    "ssh-agent",
    "ssh-keygen",
    "ssh-keyscan",
]

# Timeout per binary in seconds (0 = no timeout)
TIMEOUT = 300


def find_binary(base_dir, name):
    """Search for a binary by name under the base directory."""
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == name or f.startswith(name + "."):
                full = os.path.join(root, f)
                # Quick sanity check: is it an ELF?
                try:
                    with open(full, "rb") as fh:
                        if fh.read(4) == b"\x7fELF":
                            return full
                except:
                    pass
    return None


def clear_thumb_bit(addr):
    """Clear the lowest bit (Thumb indicator) from an ARM address."""
    return addr & ~1


def format_addr(addr):
    """Format address as 8-digit lowercase hex string without 0x prefix."""
    return f"{clear_thumb_bit(addr):08x}"


def extract_jump_tables(binary_path, binary_name, timeout=300):
    """
    Load a binary with angr, run CFGFast, and extract jump table info.

    Returns a dict matching the ground truth JSON schema.
    """
    print(f"  Loading {binary_path} ...")

    # First load to check if PIE
    proj_probe = angr.Project(binary_path, auto_load_libs=False)
    is_pie = proj_probe.loader.main_object.pic
    linked_base = proj_probe.loader.main_object.linked_base

    if is_pie:
        # Ghidra loads ARM PIE binaries at 0x10000 by default
        # Match that base so addresses align with ground truth
        GHIDRA_ARM_PIE_BASE = 0x10000
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

    # Collect jump tables from cfg.indirect_jumps
    # (NOT kb.indirect_jumps, which is empty in this angr version)
    jump_tables = []

    all_ij = cfg.indirect_jumps
    resolved_count = 0

    for addr, ij in all_ij.items():
        # Only include entries flagged as jump tables
        if not getattr(ij, "jumptable", False):
            continue

        # Only include successfully resolved ones
        targets_raw = ij.resolved_targets
        if not targets_raw:
            continue

        resolved_count += 1

        # ins_addr is the actual indirect jump instruction address
        inst_addr = clear_thumb_bit(ij.ins_addr)
        targets = sorted([clear_thumb_bit(t) for t in targets_raw])

        jump_tables.append({
            "instruction_addr": format_addr(inst_addr),
            "targets": [format_addr(t) for t in targets],
        })

    # Sort by instruction address for consistent output
    jump_tables.sort(key=lambda jt: jt["instruction_addr"])

    result = {
        "binary": binary_name,
        "jump_tables": jump_tables,
    }

    print(f"  Found {len(jump_tables)} jump tables "
          f"({len(all_ij)} indirect jumps total, "
          f"{len(all_ij) - resolved_count} unresolved/not-jumptable)")

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Allow overriding target list from command line
    targets = TARGET_BINARIES
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
        print(f"Analyzing specified binaries: {targets}")
    else:
        print(f"Analyzing default target list: {targets}")

    print(f"Base directory: {BINARY_BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    all_results = []

    for name in targets:
        print(f"[{name}]")

        # Find the binary file
        binary_path = find_binary(BINARY_BASE_DIR, name)
        if binary_path is None:
            # Also try looking for it directly
            direct = os.path.join(BINARY_BASE_DIR, name)
            if os.path.isfile(direct):
                binary_path = direct

        if binary_path is None:
            print(f"  WARNING: Binary '{name}' not found under {BINARY_BASE_DIR}")
            print(f"  Skipping.")
            print()
            continue

        print(f"  Found: {binary_path}")

        try:
            result = extract_jump_tables(binary_path, name, timeout=TIMEOUT)

            # Write individual JSON
            out_path = os.path.join(OUTPUT_DIR, f"{name}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved to {out_path}")

            all_results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        print()

    # Write combined results
    if all_results:
        combined_path = os.path.join(OUTPUT_DIR, "all_angr_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Combined results saved to {combined_path}")

    # Print summary
    print()
    print("=" * 50)
    print("Summary:")
    print("=" * 50)
    total_jt = sum(len(r["jump_tables"]) for r in all_results)
    print(f"  Binaries analyzed: {len(all_results)}/{len(targets)}")
    print(f"  Total jump tables found: {total_jt}")
    for r in all_results:
        print(f"    {r['binary']}: {len(r['jump_tables'])} jump tables")


if __name__ == "__main__":
    main()