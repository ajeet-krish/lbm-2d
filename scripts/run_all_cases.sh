#!/usr/bin/env bash
# ==============================================================
# LBM-2D: Run All Simulation Cases
# ==============================================================
# Orchestrates all Reynolds number sweeps for every case.
# Runs cylinder, cavity, and all non-cylinder cases sequentially.
#
# Usage:
#   bash scripts/run_all_cases.sh
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Build once
echo "Building LBM-2D..."
cmake -B build && cmake --build build

echo ""
echo "================================================"
echo " LBM-2D: Full Simulation Suite"
echo "================================================"
echo ""

# --- Cylinder (default case) ---
echo ">>> Cylinder Re=100 (30000 steps)"
mkdir -p output/cylinder_re100
./build/LBM_Engine 100 30000 2>&1 | tee output/cylinder_re100/run.log
echo ""

echo ">>> Cylinder Re=200 (40000 steps)"
mkdir -p output/cylinder_re200
./build/LBM_Engine 200 40000 2>&1 | tee output/cylinder_re200/run.log
echo ""

# --- Step ---
bash scripts/run_step.sh

# --- Ribs ---
bash scripts/run_ribs.sh

# --- Urban Canyon ---
bash scripts/run_urban.sh

# --- Downwash ---
bash scripts/run_downwash.sh

# --- Ahmed ---
bash scripts/run_ahmed.sh

echo ""
echo "================================================"
echo " All simulations complete."
echo "================================================"
