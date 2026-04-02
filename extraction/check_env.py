#!/usr/bin/env python3
"""Check if the environment is ready for binary analysis extraction."""

import os
import sys

print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print()

# Check angr
try:
    import angr
    print(f"[OK] angr {angr.__version__}")
except ImportError:
    print("[MISSING] angr - install with: pip install angr")

try:
    import archinfo
    arm = archinfo.ArchARM()
    print(f"[OK] archinfo {archinfo.__version__} (ARM support confirmed)")
except Exception as e:
    print(f"[ISSUE] archinfo ARM support: {e}")

try:
    import pyvex
    print(f"[OK] pyvex {pyvex.__version__}")
except ImportError:
    print("[MISSING] pyvex - should come with angr")

try:
    import cle
    print(f"[OK] cle {cle.__version__}")
except ImportError:
    print("[MISSING] cle - should come with angr")

try:
    import claripy
    print("[OK] claripy")
except ImportError:
    print("[MISSING] claripy - should come with angr")

print()

# Check data directories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
binaries_dir = os.path.join(project_root, "data", "binaries")

if os.path.isdir(binaries_dir):
    print(f"[OK] Binaries directory: {binaries_dir}")
    for entry in sorted(os.listdir(binaries_dir)):
        full = os.path.join(binaries_dir, entry)
        if os.path.isdir(full):
            files = [f for f in os.listdir(full) if not f.endswith(('.pb', '.json'))]
            print(f"     {entry}/ ({len(files)} binaries)")
else:
    print(f"[MISSING] Binaries directory: {binaries_dir}")
    print("         Run the binary copy step first (see README).")

print()
print("--- Summary ---")
try:
    import angr
    print("Environment is ready. Run: python -m extraction.extract_angr")
except ImportError:
    print("Run: pip install angr")
