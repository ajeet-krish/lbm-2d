# LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

A cache-optimized D2Q9 Lattice Boltzmann Method solver for 2D flow past
a cylinder. Validated against Williamson 1988 (Strouhal) and Tritton 1959
(drag coefficient). Includes an interactive web dashboard with real-time
canvas animation of the velocity field.

Built as an aerospace/defense portfolio piece demonstrating HPC competency
(C++20, OpenMP), CFD fundamentals, and engineering communication skills.

## Quick Start

```bash
cmake -B build && cmake --build build
./build/LBM_Engine               # Simulate Re=100 (default 30000 steps)
./build/LBM_Engine 200           # Simulate at Re=200
./build/LBM_Engine 100 12000     # Custom steps

# Run tests
./build/LBM_Tests

# Preview interactive website
python3 -m http.server -d docs 8765
open http://localhost:8765
```

## Validation (Williamson 1988 / Tritton 1959)

| Reynolds | Regime | Computed St | Literature St | Computed Cd | Literature Cd |
|----------|--------|-------------|---------------|-------------|---------------|
| 20 | Steady attached | -- | -- | 3.108 | ~2.0 |
| 40 | Steady recirculating | -- | -- | 2.293 | ~1.5 |
| 100 | Laminar shedding | 0.200 | 0.164-0.172 | 1.763 | ~1.4 |
| 200 | Laminar shedding | 0.240 | 0.180-0.195 | 1.600 | ~1.3 |

Systematic bias (~20-55%) from stair-step cylinder boundary at moderate
grid resolution (D=30 grid points). See the website discussion section
for details.

## Key Features

- **D2Q9 BGK** collision operator with Zou/He velocity inlet BC
- **Flat 1D memory layout** (std::vector) for cache-optimized access
- **OpenMP parallel** collision and streaming (collapse(2))
- **Momentum exchange** force extraction for drag/lift coefficients
- **VTK export** for ParaView visualization
- **Interactive canvas player** -- watch the flow evolve frame by frame
- **Pre-computed data** for Re=20, 40, 100, 200 at 51 frames each
- **Production-grade**: Google Test suite, GitHub Actions CI

## Interactive Website

The docs/ directory contains a 5-page portfolio website:

- **index.html** -- Home with teaser images
- **simulation.html** -- Real-time canvas animation player with play/pause,
  speed control, frame scrubber, and live Cd/Cl HUD
- **theory.html** -- LBM theory with KaTeX equations
- **implementation.html** -- Code architecture with source blocks
- **results.html** -- Validation plots, velocity field gallery, discussion

Launch with `python3 -m http.server -d docs 8765`.

## Viewing VTK Output in ParaView

```bash
# Single frame
paraview output/re100/frame_30000.vtk

# Time series
paraview output/re100/frame_*.vtk
```

In ParaView: File > Open > select VTK files > Apply. Color by
VelocityMagnitude, use the Jet colormap. Add a Glyph filter for velocity
vectors or a Stream Tracer for streamlines.

## Architecture

```
src/
  lbm_types.hpp     D2Q9 constants, equilibrium, macros, index helpers
  lbm.hpp           Core solver: collide, stream, BCs, force extraction
  main.cpp          Simulation orchestrator, CSV output
  lbm_test.cpp      Google Test unit tests (13 tests, 8 suites)

scripts/
  postprocess.py    VTK -> JSON/PNG for web viewer
  run_all_re.sh     Batch runner for Re sweep

docs/
  index.html        Home page
  simulation.html   Interactive canvas player
  theory.html       LBM theory (KaTeX)
  implementation.html  Code architecture
  results.html      Validation + field gallery
  css/style.css     CFD Jet theme
  assets/js/viewer.js   Canvas animation player
  assets/data/      Pre-computed JSON (51 frames per Re)
  assets/images/    Field renders + validation plots
```

## Extensions (Roadmap)

- [x] D2Q9 BGK with OpenMP parallelism
- [x] Drag/lift extraction with momentum exchange
- [x] Interactive web dashboard with canvas player
- [ ] MRT (Multi-Relaxation Time) collision operator
- [ ] Smagorinsky LES turbulence model
- [ ] D3Q19 / D3Q27 3D lattice
- [ ] Immersed boundary method for moving geometries

## License

MIT
