#!/usr/bin/env bash
# ==============================================================
# LBM-2D: Ribbed Channel Flow -- Reynolds Number Sweep
# ==============================================================
# Runs ribbed channel simulations at Re_H = 50, 100, 200
# Periodic in x/y, driven by body force.
# Validation: friction factor vs Webb 1971 (qualitative for laminar)
#
# Usage:
#   bash scripts/run_ribs.sh
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Build if needed
if [ ! -f build/LBM_Ribs ]; then
    echo "Building LBM_Ribs..."
    cmake -B build && cmake --build build
fi

echo ""
echo "================================================"
echo " Ribbed Channel Re Sweep"
echo "================================================"
echo ""

declare -A RE_STEPS
RE_STEPS[50]=40000
RE_STEPS[100]=50000
RE_STEPS[200]=60000

for Re in 50 100 200; do
    steps=${RE_STEPS[$Re]}
    echo "--- Re_H = $Re ($steps steps) ---"
    mkdir -p "output/ribs_re${Re}"
    ./build/LBM_Ribs "$Re" "$steps" 2>&1 | tee "output/ribs_re${Re}/run.log"
    echo ""
done

echo "================================================"
echo " All ribbed channel simulations complete."
echo "================================================"
