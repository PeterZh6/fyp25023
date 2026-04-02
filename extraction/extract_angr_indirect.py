#!/usr/bin/env python3
"""
Extract indirect control-flow sites from ARM32 ELF binaries using angr CFGFast.
Filters out PLT stubs, resolves function names, includes VEX fallback scan.
Output JSON matches Ghidra ExportIndirectFlowsCustom format.
"""

import angr
import pyvex
import json
import os
import sys
import time
import signal
import logging
import argparse

logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)
logging.getLogger("pyvex").setLevel(logging.CRITICAL)

GHIDRA_ARM_PIE_BASE = 0x10000
DEFAULT_TIMEOUT = 600


class AnalysisTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise AnalysisTimeout()


def fmt_addr(addr):
    return f"{(addr & 0xFFFFFFFE):08x}"


def _plt_ranges(proj):
    """Collect address ranges for .plt and .plt.got sections."""
    ranges = []
    obj = proj.loader.main_object
    for name in (".plt", ".plt.got"):
        sec = obj.sections_map.get(name)
        if sec is not None:
            ranges.append((sec.min_addr, sec.max_addr))
    return ranges


def _in_plt(addr, plt_ranges):
    for lo, hi in plt_ranges:
        if lo <= addr <= hi:
            return True
    return False


def _func_name(proj, addr):
    func = proj.kb.functions.floor_func(addr)
    return func.name if func else ""


def _collect_from_cfg(proj, cfg, plt_ranges):
    """Primary extraction from cfg.indirect_jumps."""
    entries = {}
    for block_addr, ij in cfg.indirect_jumps.items():
        ins_addr = getattr(ij, "ins_addr", block_addr)
        norm = ins_addr & 0xFFFFFFFE
        if _in_plt(ins_addr, plt_ranges):
            continue

        jk = getattr(ij, "jumpkind", "Ijk_Boring")
        if jk == "Ijk_Call":
            flow_type = "call"
        elif jk == "Ijk_Boring":
            flow_type = "jump"
        else:
            continue

        raw_targets = ij.resolved_targets
        if raw_targets is None:
            raw_targets = []
        elif not isinstance(raw_targets, (list, set, tuple)):
            raw_targets = list(raw_targets)
        targets = sorted(set(fmt_addr(t) for t in raw_targets))

        entries[norm] = {
            "address": fmt_addr(norm),
            "type": flow_type,
            "function": _func_name(proj, ins_addr),
            "targets": targets,
        }
    return entries


def _collect_vex_fallback(proj, cfg, plt_ranges, seen_addrs):
    """Fallback: scan CFG nodes for indirect exits missed by kb.indirect_jumps."""
    extras = {}
    for node in cfg.model.nodes():
        if node.block is None:
            continue
        try:
            block = proj.factory.block(node.addr, node.size)
            vex = block.vex
        except Exception:
            continue

        if not isinstance(vex.next, pyvex.expr.RdTmp):
            continue

        jk = vex.jumpkind
        if jk == "Ijk_Call":
            flow_type = "call"
        elif jk == "Ijk_Boring":
            flow_type = "jump"
        else:
            continue

        last_ins_addr = block.instruction_addrs[-1] if block.instruction_addrs else node.addr
        norm = last_ins_addr & 0xFFFFFFFE
        if norm in seen_addrs:
            continue
        if _in_plt(last_ins_addr, plt_ranges):
            continue

        extras[norm] = {
            "address": fmt_addr(norm),
            "type": flow_type,
            "function": _func_name(proj, last_ins_addr),
            "targets": [],
        }
    return extras


def extract_indirect_flows(binary_path, binary_name, timeout=DEFAULT_TIMEOUT):
    proj_probe = angr.Project(binary_path, auto_load_libs=False)
    is_pie = proj_probe.loader.main_object.pic

    if is_pie:
        proj = angr.Project(binary_path, auto_load_libs=False,
                            main_opts={"base_addr": GHIDRA_ARM_PIE_BASE})
        print(f"  PIE binary, rebased to 0x{GHIDRA_ARM_PIE_BASE:x}")
    else:
        proj = proj_probe
        base = proj.loader.main_object.linked_base
        print(f"  Non-PIE, linked base 0x{base:x}")

    plt_ranges = _plt_ranges(proj)
    if plt_ranges:
        print(f"  PLT ranges: {', '.join(f'0x{lo:x}-0x{hi:x}' for lo, hi in plt_ranges)}")

    prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        print(f"  Running CFGFast (timeout={timeout}s) ...")
        t0 = time.time()
        cfg = proj.analyses.CFGFast(normalize=True, resolve_indirect_jumps=True)
        elapsed = time.time() - t0
        print(f"  CFGFast done in {elapsed:.1f}s")

        entries = _collect_from_cfg(proj, cfg, plt_ranges)
        kb_count = len(entries)

        extras = _collect_vex_fallback(proj, cfg, plt_ranges, set(entries.keys()))
        entries.update(extras)
        vex_count = len(extras)
    except AnalysisTimeout:
        print(f"  TIMEOUT after {timeout}s")
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)

    flows = sorted(entries.values(), key=lambda e: e["address"])

    jumps = sum(1 for f in flows if f["type"] == "jump")
    calls = sum(1 for f in flows if f["type"] == "call")
    resolved = sum(1 for f in flows if f["targets"])
    unresolved = len(flows) - resolved

    print(f"  Total: {len(flows)} indirect flows "
          f"(jump={jumps}, call={calls}, resolved={resolved}, unresolved={unresolved})")
    print(f"  Sources: kb.indirect_jumps={kb_count}, vex_fallback={vex_count}")

    arch_name = proj.arch.name
    return {
        "binary": binary_name,
        "arch": arch_name,
        "indirect_flows": flows,
    }


def find_binaries(directory):
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

    parser = argparse.ArgumentParser(
        description="Extract indirect control-flow sites via angr (PLT-filtered)")
    parser.add_argument("-d", "--binary-dir",
                        default=os.path.join(project_root, "data", "binaries", "openssl_gcc_O3"))
    parser.add_argument("-o", "--output-dir",
                        default=os.path.join(project_root, "data", "angr_results"))
    parser.add_argument("-t", "--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Per-binary timeout in seconds (default: 600)")
    parser.add_argument("binaries", nargs="*",
                        help="Specific binary names (default: all ELF in directory)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_binaries = find_binaries(args.binary_dir)
    if args.binaries:
        name_set = set(args.binaries)
        all_binaries = [(n, p) for n, p in all_binaries if n in name_set]

    print(f"Binary dir:  {args.binary_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Binaries:    {len(all_binaries)}")
    print(f"Timeout:     {args.timeout}s")
    print()

    results = []
    for name, path in all_binaries:
        print(f"[{name}]")
        try:
            result = extract_indirect_flows(path, name, timeout=args.timeout)
            if result is None:
                print(f"  Skipped (timeout)")
                print()
                continue
            out_path = os.path.join(args.output_dir, f"{name}_angr.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved to {out_path}")
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 50)
    print("Summary")
    print("=" * 50)
    total_flows = sum(len(r["indirect_flows"]) for r in results)
    print(f"  Binaries analyzed: {len(results)}/{len(all_binaries)}")
    print(f"  Total indirect flows: {total_flows}")
    for r in results:
        flows = r["indirect_flows"]
        n = len(flows)
        j = sum(1 for f in flows if f["type"] == "jump")
        c = sum(1 for f in flows if f["type"] == "call")
        res = sum(1 for f in flows if f["targets"])
        print(f"    {r['binary']}: {n} flows (jump={j}, call={c}, resolved={res})")


if __name__ == "__main__":
    main()
