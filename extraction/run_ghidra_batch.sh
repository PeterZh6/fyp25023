#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY_DIR="${1:-$PROJECT_ROOT/data/binaries/ssh_servers_gcc_O3}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/data/ghidra_results}"
GHIDRA_SCRIPTS="$SCRIPT_DIR/ghidra_scripts"
GHIDRA_PROJ="/tmp/ghidra_proj"

mkdir -p "$OUTPUT_DIR" "$GHIDRA_PROJ"

echo "Binary dir:  $BINARY_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Script dir:  $GHIDRA_SCRIPTS"
echo

for binary in "$BINARY_DIR"/*; do
    if [ -f "$binary" ] && file "$binary" | grep -q "ELF"; then
        name=$(basename "$binary")
        echo "Processing $name..."

        analyzeHeadless "$GHIDRA_PROJ" "proj_$name" \
            -import "$binary" \
            -scriptPath "$GHIDRA_SCRIPTS" \
            -postScript ExportIndirectFlowsCustom.java "$OUTPUT_DIR" \
            -deleteProject \
            2>/dev/null

        echo "  Done."
    fi
done

echo
echo "Results in $OUTPUT_DIR/"
