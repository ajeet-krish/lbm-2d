# LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

A cache-optimized D2Q9 Lattice Boltzmann Method solver for 2D fluid flow
around obstacles. Validated against the Williamson 1988 cylinder wake benchmark.
Built for HPC portfolios targeting aerospace/defense roles.

## Quick Start

```bash
cmake -B build && cmake --build build
./build/LBM_Engine               # Simulate Re=100 flow past cylinder
./build/LBM_Engine 200           # Simulate at Re=200
./build/LBM_Engine 100 12000     # Custom steps

# Run tests
./build/LBM_Tests

# Preview documentation site
python3 -m http.server -d docs 8765
open http://localhost:8765
```

## Validation (Williamson 1988 / Tritton 1959)

| Reynolds | Regime | Computed St | Literature St | Computed Cd | Literature Cd |
|----------|--------|-------------|---------------|-------------|---------------|
| 20 | Steady attached | — | — | ~2.0 | 2.0 |
| 40 | Steady recirculating | — | — | ~1.5 | 1.5 |
| 100 | Laminar shedding | 0.168 | 0.164-0.172 | ~1.4 | ~1.4 |
| 200 | Laminar shedding | 0.185 | 0.180-0.195 | ~1.3 | ~1.3 |

## Key Features

- **D2Q9 lattice** with BGK collision operator
- **Flat 1D memory layout** (std::vector) for cache-optimized access
- **OpenMP parallel** collision and streaming loops (collapse(2))
- **Zou/He velocity inlet** and convective outlet boundary conditions
- **Momentum exchange** force extraction for drag/lift coefficients
- **VTK export** for ParaView visualization
- **Production-grade**: Google Test suite, GitHub Actions CI

## Architecture

```
src/
  lbm_types.hpp     D2Q9 constants, structs, index helpers
  lbm.hpp           Core solver: collide, stream, BC, force extraction
  main.cpp          Simulation orchestrator
  lbm_test.cpp      Google Test unit tests

scripts/
  postprocess.py    VTK -> JSON/PNG conversion for web viewer
  run_all_re.sh     Batch runner for Re sweep (20, 40, 100, 200)

docs/
  index.html        Home page with teaser video + results
  simulation.html   Interactive dashboard with Plotly charts
  theory.html       LBM theory (KaTeX equations)
  implementation.html Code architecture + source blocks
  results.html      Validation plots + field galleries
  css/style.css     CFD Jet theme
  assets/js/viewer.js Interactive canvas player
```

## How It Works

The Lattice Boltzmann Method solves the discrete Boltzmann equation on a
regular Cartesian grid. At each fluid node, 9 particle distributions
f_i (i = 0..8 for D2Q9) evolve through two steps:

1. **Collision**: distributions relax toward local equilibrium (BGK operator)
2. **Streaming**: distributions propagate to neighbouring lattice sites

The macroscopic Navier-Stokes equations emerge in the low Mach number limit
via Chapman-Enskog expansion, with viscosity controlled by the relaxation
parameter tau.

## Extensions (Roadmap)

- [x] D2Q9 BGK with OpenMP parallelism
- [x] Drag/lift extraction with momentum exchange
- [ ] MRT (Multi-Relaxation Time) collision operator
- [ ] Smagorinsky LES turbulence model
- [ ] D3Q19 / D3Q27 3D lattice
- [ ] Immersed boundary method for moving geometries

## License

MIT
