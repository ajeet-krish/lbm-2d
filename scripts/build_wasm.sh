#!/bin/bash
# ==========================================================================
# LBM-2D Emscripten WASM build script
# ==========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src"
OUT_DIR="$PROJECT_DIR/docs/assets/wasm"

# Use Homebrew Python 3.14 for Emscripten
export EMSDK_PYTHON="/opt/homebrew/opt/python@3.14/bin/python3"

EMCC="/opt/homebrew/bin/emcc"

echo "Building LBM-2D WASM solver..."
echo "  Source: $SRC_DIR/wasm_main.cpp"
echo "  Output: $OUT_DIR/lbm_solver.wasm"

mkdir -p "$OUT_DIR"

"$EMCC" "$SRC_DIR/wasm_main.cpp" \
    -o "$OUT_DIR/lbm_solver.js" \
    -s WASM=1 \
    -s EXPORTED_FUNCTIONS='["_wasm_init","_wasm_set_shape","_wasm_set_tau","_wasm_step","_wasm_get_cd","_wasm_get_cl","_wasm_get_f_ptr","_wasm_get_obstacle_ptr","_wasm_get_u_ptr","_wasm_get_v_ptr","_wasm_get_rho_ptr","_wasm_get_nx","_wasm_get_ny","_wasm_get_step","_malloc","_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAP32"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s TOTAL_STACK=524288 \
    -O2 \
    -std=c++20 \
    --no-entry \
    -Wno-invalid-offsetof \
    -flto

echo ""
echo "Build complete!"
echo "  $OUT_DIR/lbm_solver.wasm"
echo "  $OUT_DIR/lbm_solver.js"
echo ""
echo "Size:"
ls -lh "$OUT_DIR/lbm_solver.wasm" "$OUT_DIR/lbm_solver.js"
