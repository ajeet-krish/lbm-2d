#!/usr/bin/env bash
# Batch runner: lid-driven cavity sweep
# Usage: bash scripts/run_cavity.sh
set -euo pipefail

NX="${1:-128}"  # grid size (square, same for X and Y)

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RESET='\033[0m'

echo -e "${CYAN}============================================${RESET}"
echo -e "${CYAN} LBM-2D: Lid-Driven Cavity Batch Runner${RESET}"
echo -e "${CYAN} Grid: ${NX}x${NX}${RESET}"
echo -e "${CYAN}============================================${RESET}"

cmake -B build 2>&1 | tail -1
cmake --build build 2>&1 | tail -1

declare -A STEPS
STEPS[100]=20000
STEPS[400]=30000
STEPS[1000]=40000

for RE in 100 400 1000; do
    echo -e "\n${GREEN}Running Re = ${RE}  (${STEPS[$RE]} steps, ${NX}x${NX})${RESET}"
    mkdir -p "output/cavity/re${RE}"

    START=$(date +%s)
    ./build/LBM_Cavity "$RE" "$NX" "${STEPS[$RE]}"
    END=$(date +%s)

    DURATION=$((END - START))
    echo -e "${GREEN}Re = ${RE} completed in ${DURATION}s${RESET}"
done

echo -e "\n${CYAN}============================================${RESET}"
echo -e "${CYAN} All cavity simulations complete.${RESET}"
echo -e "${CYAN}============================================${RESET}"
