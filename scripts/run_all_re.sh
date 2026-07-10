#!/usr/bin/env bash
# Batch runner for LBM-2D Reynolds number sweep
# Usage: bash scripts/run_all_re.sh
set -euo pipefail

echo "=== LBM-2D Batch Runner ==="

# Build if needed
if [ ! -f "build/LBM_Engine" ]; then
    echo "Building solver..."
    cmake -B build && cmake --build build
fi

# Reynolds numbers to simulate
RE_VALUES=(20 40 100 200)

# Step counts tuned for each Re:
#   Steady flows (Re=20,40): need 20000 steps to reach steady state
#   Unsteady (Re=100): shedding develops by ~18000 steps
#   Unsteady (Re=200): shedding develops by ~12000 steps
STEPS_LIST=(20000 20000 30000 20000)

for i in "${!RE_VALUES[@]}"; do
    RE="${RE_VALUES[$i]}"
    STEPS="${STEPS_LIST[$i]}"
    echo ""
    echo "--- Simulating Re = ${RE} (${STEPS} steps) ---"
    ./build/LBM_Engine "${RE}" "${STEPS}"
    echo "--- Done Re = ${RE} ---"
done

echo ""
echo "=== All simulations complete ==="
echo "Output directories:"
for RE in "${RE_VALUES[@]}"; do
    FRAMES=$(ls output/re${RE}/frame_*.vtk 2>/dev/null | wc -l)
    echo "  output/re${RE}/  (${FRAMES} VTK frames)"
done

# Optional: post-process to JSON
echo ""
echo "To convert to JSON for web viewer, run:"
for RE in "${RE_VALUES[@]}"; do
    echo "  python3 scripts/postprocess.py output/re${RE} --json --every 5"
done
