#!/usr/bin/env python3
"""
Measure wall-clock analysis cost of Ghidra and angr per binary,
compute amortized per-site cost, and output L1:L2 cost ratio.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from statistics import median

from calibration.config import filter_binary_entries

PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

GHIDRA_HEADLESS = os.path.join(
    PROJECT_ROOT, "tools", "ghidra_12.0", "support", "analyzeHeadless"
)
GHIDRA_SCRIPTS = os.path.join(PROJECT_ROOT, "extraction", "ghidra_scripts")
ANGR_SCRIPT = os.path.join(PROJECT_ROOT, "extraction", "extract_angr_indirect.py")

SILVER_LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "silver_labels")
BINARIES_DIR = os.path.join(PROJECT_ROOT, "data", "binaries")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "data", "calibration", "cost_benchmark.json")

TIMEOUT_SENTINEL = "TIMEOUT"


def load_silver_labels():
    """Scan silver_labels/ for *_labels.json; return list of (binary_name, site_count)."""
    entries = []
    for path in sorted(glob.glob(os.path.join(SILVER_LABELS_DIR, "*_labels.json"))):
        with open(path) as f:
            data = json.load(f)
        binary_name = data.get("binary", "")
        total_sites = data.get("total", 0)
        if not binary_name:
            continue
        entries.append((binary_name, total_sites, path))
    return entries


def build_binary_map():
    """Walk data/binaries/ subdirs and map filename -> absolute path for ELF files."""
    bmap = {}
    for subdir in sorted(os.listdir(BINARIES_DIR)):
        dirpath = os.path.join(BINARIES_DIR, subdir)
        if not os.path.isdir(dirpath):
            continue
        for fname in os.listdir(dirpath):
            fpath = os.path.join(dirpath, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "rb") as f:
                    if f.read(4) == b"\x7fELF":
                        bmap[fname] = fpath
            except (PermissionError, IsADirectoryError):
                pass
    return bmap


def ghidra_command(binary_path, binary_name, output_dir):
    proj_dir = f"/tmp/ghidra_bench_{binary_name}"
    return (
        [
            GHIDRA_HEADLESS,
            proj_dir,
            f"proj_{binary_name}",
            "-import", binary_path,
            "-scriptPath", GHIDRA_SCRIPTS,
            "-postScript", "ExportIndirectFlowsCustom.java", output_dir,
            "-deleteProject",
        ],
        proj_dir,
    )


def angr_command(binary_path, output_dir, angr_timeout):
    binary_dir = os.path.dirname(binary_path)
    binary_name = os.path.basename(binary_path)
    return [
        sys.executable,
        ANGR_SCRIPT,
        "--binary-dir", binary_dir,
        "--output-dir", output_dir,
        "--timeout", str(angr_timeout),
        binary_name,
    ]


def run_timed(cmd, timeout, label=""):
    """Run cmd with wall-clock timing. Returns elapsed seconds or TIMEOUT_SENTINEL."""
    t0 = time.time()
    try:
        subprocess.run(
            cmd,
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return round(time.time() - t0, 1)
    except subprocess.TimeoutExpired:
        return TIMEOUT_SENTINEL
    except Exception as e:
        print(f"  [{label}] ERROR: {e}", file=sys.stderr)
        return TIMEOUT_SENTINEL


def fmt_time(val):
    if val is None:
        return "-"
    if val == TIMEOUT_SENTINEL:
        return "TIMEOUT"
    return f"{val:.1f}"


def fmt_cost(val):
    if val is None:
        return "-"
    return f"{val:.2f}"


def print_header():
    hdr = (
        f"{'Binary':<45} {'Sites':>5}  "
        f"{'Ghidra(s)':>10} {'angr(s)':>10}  "
        f"{'Ghidra/site':>11} {'angr/site':>11}"
    )
    print(hdr)
    print("-" * len(hdr))


def print_row(name, sites, ghidra_t, angr_t, ghidra_ps, angr_ps):
    print(
        f"{name:<45} {sites:>5}  "
        f"{fmt_time(ghidra_t):>10} {fmt_time(angr_t):>10}  "
        f"{fmt_cost(ghidra_ps):>11} {fmt_cost(angr_ps):>11}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ghidra vs angr analysis cost")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-tool timeout in seconds (default: 600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-ghidra", action="store_true",
                        help="Skip Ghidra benchmarking")
    parser.add_argument("--skip-angr", action="store_true",
                        help="Skip angr benchmarking")
    args = parser.parse_args()

    labels = load_silver_labels()
    labels = filter_binary_entries(labels, lambda entry: entry[0])
    bmap = build_binary_map()

    tasks = []
    for binary_name, sites, label_path in labels:
        if binary_name not in bmap:
            print(f"WARNING: no ELF found for '{binary_name}', skipping", file=sys.stderr)
            continue
        tasks.append((binary_name, sites, bmap[binary_name]))

    tasks.sort(key=lambda t: t[1])

    if not tasks:
        print("No binaries to benchmark.", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking {len(tasks)} binaries (timeout={args.timeout}s)")
    print(f"  skip-ghidra={args.skip_ghidra}, skip-angr={args.skip_angr}")
    print()

    if args.dry_run:
        for binary_name, sites, binary_path in tasks:
            print(f"[{binary_name}] sites={sites}")
            if not args.skip_ghidra:
                cmd, proj_dir = ghidra_command(binary_path, binary_name, "/tmp/_dry")
                print(f"  Ghidra: {' '.join(cmd)}")
            if not args.skip_angr:
                cmd = angr_command(binary_path, "/tmp/_dry", args.timeout)
                print(f"  angr:   {' '.join(cmd)}")
            print()
        return

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    results = []
    print_header()

    for binary_name, sites, binary_path in tasks:
        ghidra_time = None
        angr_time = None
        ghidra_per_site = None
        angr_per_site = None

        # --- Ghidra ---
        if not args.skip_ghidra:
            tmpdir_ghidra = tempfile.mkdtemp(prefix="ghidra_out_")
            cmd, proj_dir = ghidra_command(binary_path, binary_name, tmpdir_ghidra)
            if os.path.exists(proj_dir):
                shutil.rmtree(proj_dir, ignore_errors=True)
            os.makedirs(proj_dir, exist_ok=True)
            ghidra_time = run_timed(cmd, timeout=args.timeout, label=f"ghidra/{binary_name}")
            shutil.rmtree(tmpdir_ghidra, ignore_errors=True)
            shutil.rmtree(proj_dir, ignore_errors=True)
            if ghidra_time != TIMEOUT_SENTINEL and sites > 0:
                ghidra_per_site = ghidra_time / sites

        # --- angr ---
        if not args.skip_angr:
            tmpdir_angr = tempfile.mkdtemp(prefix="angr_out_")
            angr_timeout_internal = args.timeout
            angr_timeout_subprocess = args.timeout + 60
            cmd = angr_command(binary_path, tmpdir_angr, angr_timeout_internal)
            angr_time = run_timed(cmd, timeout=angr_timeout_subprocess, label=f"angr/{binary_name}")
            shutil.rmtree(tmpdir_angr, ignore_errors=True)
            if angr_time != TIMEOUT_SENTINEL and sites > 0:
                angr_per_site = angr_time / sites

        print_row(binary_name, sites, ghidra_time, angr_time, ghidra_per_site, angr_per_site)

        results.append({
            "binary": binary_name,
            "sites": sites,
            "binary_path": binary_path,
            "ghidra_time": ghidra_time,
            "angr_time": angr_time,
            "ghidra_per_site": ghidra_per_site,
            "angr_per_site": angr_per_site,
        })

    # --- Summary ---
    print()
    ghidra_costs = [r["ghidra_per_site"] for r in results
                    if r["ghidra_per_site"] is not None]
    angr_costs = [r["angr_per_site"] for r in results
                  if r["angr_per_site"] is not None]

    summary = {}

    if ghidra_costs:
        l1_median = median(ghidra_costs)
        summary["ghidra_per_site_median"] = round(l1_median, 4)
        print(f"Ghidra (L1) per-site median: {l1_median:.4f}s  (n={len(ghidra_costs)})")
    else:
        l1_median = None
        print("Ghidra (L1): no valid data")

    if angr_costs:
        l2_median = median(angr_costs)
        summary["angr_per_site_median"] = round(l2_median, 4)
        print(f"angr   (L2) per-site median: {l2_median:.4f}s  (n={len(angr_costs)})")
    else:
        l2_median = None
        print("angr   (L2): no valid data")

    if l1_median is not None and l2_median is not None and l1_median > 0:
        ratio = l2_median / l1_median
        summary["l1_l2_ratio"] = round(ratio, 2)
        print(f"L1:L2 = 1:{ratio:.2f}")
    else:
        summary["l1_l2_ratio"] = None
        print("L1:L2 ratio: N/A")

    # --- Write JSON ---
    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timeout": args.timeout,
        "skip_ghidra": args.skip_ghidra,
        "skip_angr": args.skip_angr,
        "summary": summary,
        "benchmarks": results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
