#!/usr/bin/env bash
# ==============================================================
# LBM-2D: Ahmed Body (2D Slice) -- Slant Angle Sweep
# ==============================================================
# Runs Ahmed body simulations at Re_H = 1000 with slant angles
# 20, 25, 30, 35 degrees.
# Validation: Cd(alpha) trend vs Ahmed 1984 (qualitative for 2D)
#
# Usage:
#   bash scripts/run_ahmed.sh
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Build if needed
if [ ! -f build/LBM_Ahmed ]; then
    echo "Building LBM_Ahmed..."
    cmake -B build && cmake --build build
fi

echo ""
echo "================================================"
echo " Ahmed Body Slant Sweep (Re = 1000)"
echo "================================================"
echo ""

for slant in 20 25 30 35; do
    echo "--- Re_H = 1000, slant = ${slant} deg ---"
    mkdir -p "output/ahmed_re1000_slant${slant}"
    ./build/LBM_Ahmed 1000 "$slant" 2>&1 | tee "output/ahmed_re1000_slant${slant}/run.log"
    echo ""
done

echo "================================================"
echo " All Ahmed body simulations complete."
echo "================================================"
