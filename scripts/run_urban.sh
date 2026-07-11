#!/usr/bin/env bash
# ==============================================================
# LBM-2D: Urban Canyon Flow -- Reynolds Number Sweep
# ==============================================================
# Runs urban canyon simulations at Re_H = 100, 200
# Two-building street canyon, 1:1 aspect ratio.
# Validation: flow regime vs Oke 1988
#
# Usage:
#   bash scripts/run_urban.sh
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Build if needed
if [ ! -f build/LBM_UrbanCanyon ]; then
    echo "Building LBM_UrbanCanyon..."
    cmake -B build && cmake --build build
fi

echo ""
echo "================================================"
echo " Urban Canyon Re Sweep"
echo "================================================"
echo ""

declare -A RE_STEPS
RE_STEPS[100]=40000
RE_STEPS[200]=50000

for Re in 100 200; do
    steps=${RE_STEPS[$Re]}
    echo "--- Re_H = $Re ($steps steps) ---"
    mkdir -p "output/urban_re${Re}"
    ./build/LBM_UrbanCanyon "$Re" "$steps" 2>&1 | tee "output/urban_re${Re}/run.log"
    echo ""
done

echo "================================================"
echo " All urban canyon simulations complete."
echo "================================================"
