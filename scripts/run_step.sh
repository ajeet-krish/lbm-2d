#!/usr/bin/env bash
# ==============================================================
# LBM-2D: Backward-Facing Step -- Reynolds Number Sweep
# ==============================================================
# Runs step simulations at Re_H = 100, 200, 400
# Validates Xr/H reattachment length against Armaly et al. 1983
#
# Usage:
#   bash scripts/run_step.sh
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Build if needed
if [ ! -f build/LBM_Step ]; then
    echo "Building LBM_Step..."
    cmake -B build && cmake --build build
fi

echo ""
echo "================================================"
echo " Backward-Facing Step Re Sweep"
echo "================================================"
echo ""

declare -A RE_STEPS
RE_STEPS[100]=30000
RE_STEPS[200]=40000
RE_STEPS[400]=50000

for Re in 100 200 400; do
    steps=${RE_STEPS[$Re]}
    echo "--- Re_H = $Re ($steps steps) ---"
    mkdir -p "output/step_re${Re}"
    ./build/LBM_Step "$Re" "$steps" 2>&1 | tee "output/step_re${Re}/run.log"
    echo ""
done

echo "================================================"
echo " All step simulations complete."
echo "================================================"
