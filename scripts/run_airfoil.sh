#!/bin/bash
# ==========================================================================
# Batch runner: NACA 4-digit airfoil analysis
# ==========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "Building LBM_Airfoil..."
cmake -B "$BUILD_DIR" && cmake --build "$BUILD_DIR" --target LBM_Airfoil

echo ""
echo "=========================================="
echo " NACA 0012 (symmetric) AoA sweep at Re=1000"
echo "=========================================="
for aoa in 0 4 8 12 16; do
    echo ""
    echo "--- Running NACA 0012, Re=1000, AoA=${aoa} ---"
    "$BUILD_DIR/LBM_Airfoil" 0012 1000 "$aoa" 30000
done

echo ""
echo "=========================================="
echo " NACA 2412 (cambered 2%) AoA sweep at Re=1000"
echo "=========================================="
for aoa in 0 4 8; do
    echo ""
    echo "--- Running NACA 2412, Re=1000, AoA=${aoa} ---"
    "$BUILD_DIR/LBM_Airfoil" 2412 1000 "$aoa" 30000
done

echo ""
echo "All airfoil simulations complete."
echo "Output in: $PROJECT_DIR/output/airfoil/"
