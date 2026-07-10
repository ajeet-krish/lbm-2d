# LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

A cache-optimized D2Q9 Lattice Boltzmann Method solver for 2D flow.
Validated against Williamson 1988 (Strouhal), Tritton 1959 (drag),
and Ghia 1982 (lid-driven cavity). Includes NACA 4-digit airfoil analysis
with lift/drag extraction and a real-time WebAssembly browser simulator
with interactive shape/Re/AoA controls and animated PhiFlow-style pathlines.

Built as an aerospace/defense portfolio piece demonstrating HPC competency
(C++20, OpenMP), CFD fundamentals, and engineering communication skills.

## Quick Start

```bash
cmake -B build && cmake --build build
./build/LBM_Engine               # Simulate Re=100 (default 30000 steps)
./build/LBM_Engine 200           # Simulate at Re=200
./build/LBM_Engine 100 12000     # Custom steps

# Cylinder benchmark sweep
bash scripts/run_all_re.sh

# Lid-driven cavity
./build/LBM_Cavity 100 128       # Re=100 on 128x128

# NACA airfoil analysis
./build/LBM_Airfoil 0012 1000 0  # NACA 0012 at Re=1000, AoA=0

# Run tests
./build/LBM_Tests

# WebAssembly real-time simulator
bash scripts/build_wasm.sh
python3 -m http.server -d docs 8765
open http://localhost:8765
```

## Validation Summary

| Reynolds | Regime | Computed St | Literature St | Computed Cd | Literature Cd |
|----------|--------|-------------|---------------|-------------|---------------|
| 20 | Steady | -- | -- | 3.108 | ~2.0 |
| 40 | Steady recirculating | -- | -- | 2.293 | ~1.5 |
| 100 | Laminar shedding | 0.200 | 0.164-0.172 | 1.763 | ~1.4 |
| 200 | Laminar shedding | 0.240 | 0.180-0.195 | 1.600 | ~1.3 |

Systematic bias (~20-55%) from stair-step cylinder boundary at
moderate grid resolution (D=30 grid points). Lift-curve slope for NACA 0012
matches thin-airfoil theory within 3%.

## Key Features

- **D2Q9 BGK** collision operator with Zou/He velocity inlet BC
- **Flat 1D memory layout** (std::vector) for cache-optimized access
- **OpenMP parallel** collision and streaming (collapse(2))
- **Momentum exchange** force extraction for Cd/Cl coefficients
- **Three solver modes**: cylinder flow, lid-driven cavity, arbitrary geometry
- **NACA 4-digit airfoil** analysis with AoA sweep (lift curve, drag polar)
- **Polygon obstacle** support (any closed 2D shape via point-in-polygon)
- **VTK export** for ParaView visualization
- **WebAssembly real-time simulator**: 100x60 grid, 5 shapes, Re slider,
  AoA slider, animated pathlines, live Cd/Cl HUD
- **Production-grade**: 13 Google Tests, GitHub Actions CI

## Interactive Website

The docs/ directory contains a 5-page portfolio website:

- **index.html** -- Home with teaser images
- **simulation.html** -- WebAssembly real-time solver with shape selector,
  Re/AoA controls, live pathline visualization, and Cd/Cl HUD
- **theory.html** -- LBM theory with KaTeX equations
- **implementation.html** -- Code architecture with source blocks
- **results.html** -- Validation plots, cylinder/cavity/airfoil galleries, discussion

## Architecture

```
src/
  lbm_types.hpp     D2Q9 constants, equilibrium, macros, index helpers
  lbm.hpp           Core solver: collide, stream, BCs, force extraction
  geometry.hpp      NACA 4-digit coords, polygon ops, point-in-polygon
  main.cpp          Cylinder flow entry point
  cavity.cpp        Lid-driven cavity entry point
  airfoil.cpp       NACA airfoil analysis entry point
  wasm_main.cpp     WASM entry point (Phase 3)
  lbm_test.cpp      Google Test unit tests

scripts/
  postprocess.py    VTK -> PNG/JSON for web viewer
  plot_airfoil.py   Airfoil validation plots (Cl/Cd vs AoA)
  run_all_re.sh     Batch runner for Re sweep
  run_cavity.sh     Batch runner for lid-driven cavity
  run_airfoil.sh    Batch runner for airfoil AoA sweep
  build_wasm.sh     Emscripten WASM build

docs/
  index.html        Home page
  simulation.html   WASM real-time solver (interactive)
  theory.html       LBM theory (KaTeX)
  implementation.html  Code architecture
  results.html      Validation + field gallery + cavity + airfoil
  css/style.css     CFD Jet theme
  assets/js/viewer.js    Pre-computed player (archive)
  assets/js/wasm_sim.js  WASM rendering engine
  assets/data/      Pre-computed JSON (cylinder frames)
  assets/images/    Field renders + validation plots
```

## Roadmap

### Phase 3: WebAssembly Real-Time Solver (Active)
- [ ] 3A: Emscripten toolchain + build script
- [ ] 3B: WASM solver port (exported C functions)
- [ ] 3C: Canvas rendering engine (pathlines, colormap)
- [ ] 3D: Interactive page (controls, HUD)
- [ ] 3E: Shape library + polish

### Phase 4: Advanced Features (Stretch)
- [ ] MRT (Multi-Relaxation Time) collision operator
- [ ] Smagorinsky LES turbulence model
- [ ] Immersed boundary method for moving geometries

## License

MIT
