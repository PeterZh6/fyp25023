#!/usr/bin/env python3
"""Check if the environment is ready for angr jump table extraction."""

import sys
import shutil

print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print()

# Check angr
try:
    import angr
    print(f"[OK] angr {angr.__version__}")
except ImportError:
    print("[MISSING] angr - install with: pip install angr")

# Check archinfo (comes with angr, but verify ARM support)
try:
    import archinfo
    arm = archinfo.ArchARM()
    print(f"[OK] archinfo {archinfo.__version__} (ARM support confirmed)")
except Exception as e:
    print(f"[ISSUE] archinfo ARM support: {e}")

# Check pyvex (VEX IR lifter, needed for CFG analysis)
try:
    import pyvex
    print(f"[OK] pyvex {pyvex.__version__}")
except ImportError:
    print("[MISSING] pyvex - should come with angr")

# Check cle (binary loader)
try:
    import cle
    print(f"[OK] cle {cle.__version__}")
except ImportError:
    print("[MISSING] cle - should come with angr")

# Check claripy (constraint solver)
try:
    import claripy
    print(f"[OK] claripy")
except ImportError:
    print("[MISSING] claripy - should come with angr")

# Check json (stdlib, should always be there)
try:
    import json
    print("[OK] json (stdlib)")
except ImportError:
    print("[MISSING] json - this shouldn't happen")

print()

# Check if target directory exists
import os
target_dir = os.path.expanduser("/home/peterzh/sec/playground/rl/extract_gt/gts/")
if os.path.isdir(target_dir):
    print(f"[OK] Target directory exists: {target_dir}")
    # List subdirs
    entries = sorted(os.listdir(target_dir))
    print(f"     Found {len(entries)} entries:")
    for e in entries[:15]:
        full = os.path.join(target_dir, e)
        if os.path.isdir(full):
            files = os.listdir(full)
            print(f"       {e}/ ({len(files)} files)")
        else:
            print(f"       {e}")
    if len(entries) > 15:
        print(f"       ... and {len(entries) - 15} more")
else:
    print(f"[INFO] Target directory not found: {target_dir}")
    print("       Update the path in extract_jump_tables.py before running.")

print()
print("--- Summary ---")
try:
    import angr
    print("Environment is ready. You can run extract_jump_tables.py.")
except ImportError:
    print("Run: pip install angr")
    print("Then re-run this check.")
