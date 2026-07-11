#!/usr/bin/env bash
# ==============================================================
# LBM-2D: Building Downwash -- Reynolds Number Sweep
# ==============================================================
# Runs building downwash simulations at Re_H = 100, 200
# Tall building upstream, low-rise downstream.
# Validation: Cp distribution vs Hunt 1984 (qualitative)
#
# Usage:
#   bash scripts/run_downwash.sh
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Build if needed
if [ ! -f build/LBM_Downwash ]; then
    echo "Building LBM_Downwash..."
    cmake -B build && cmake --build build
fi

echo ""
echo "================================================"
echo " Building Downwash Re Sweep"
echo "================================================"
echo ""

declare -A RE_STEPS
RE_STEPS[100]=40000
RE_STEPS[200]=50000

for Re in 100 200; do
    steps=${RE_STEPS[$Re]}
    echo "--- Re_H = $Re ($steps steps) ---"
    mkdir -p "output/downwash_re${Re}"
    ./build/LBM_Downwash "$Re" "$steps" 2>&1 | tee "output/downwash_re${Re}/run.log"
    echo ""
done

echo "================================================"
echo " All downwash simulations complete."
echo "================================================"
