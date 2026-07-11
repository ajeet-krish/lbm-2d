# LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

A cache-optimized D2Q9 Lattice Boltzmann Method solver for 2D flow.
Features a **Multi-Relaxation Time (MRT) collision operator** (default,
BGK fallback), Bouzidi interpolated bounce-back for curved boundaries, and
direct JSON output pipeline. Validated against Williamson 1988 (Strouhal),
Tritton 1959 (drag), Ghia 1982 (lid-driven cavity), Armaly 1983
(backward-facing step), Oke 1988 (urban canyon), and more.

Built as an aerospace/defense portfolio piece demonstrating HPC competency
(C++20, OpenMP), CFD fundamentals (MRT-LBM, Bouzidi, momentum exchange),
and engineering communication skills (interactive web results with
per-case dedicated pages and comparison sliders).

## Quick Start

```bash
cmake -B build && cmake --build build
./build/LBM_Engine               # Simulate cylinder Re=100 (JSON output)

# Fluid analysis cases
./build/LBM_Step 100             # Backward-facing step at Re=100
./build/LBM_Ribs 100             # Ribbed channel at Re=100
./build/LBM_UrbanCanyon 100      # Urban canyon (top-down) at Re=100
./build/LBM_UrbanCanyon --mode side --ar 0.5  # Side-view canyon
./build/LBM_Downwash 100         # Building downwash at Re=100
./build/LBM_Ahmed 1000 30        # Ahmed body at Re=1000, slant=30 deg

# Run tests
./build/LBM_Tests

# Post-process (separate contour + streamline PNGs)
python3 scripts/postprocess.py output/step_re100 --split --cmap coolwarm
python3 scripts/postprocess.py output/step_re100 --strouhal

# Preview website
python3 -m http.server -d docs 8765
open http://localhost:8765
```

## Validation Coverage

| Case | Re Range | Key Metric | Literature |
|------|----------|-----------|------------|
| Cylinder wake | 20-200 | St, Cd | Williamson 1988, Tritton 1959 |
| Lid-driven cavity | 100-1000 | u-profile | Ghia, Ghia & Shin 1982 |
| NACA 4-digit airfoil | Re=1000, AoA 0-16 deg | Cl, Cd | Thin-airfoil theory |
| Backward-facing step | 100-400 | Xr/H | Armaly et al. 1983 |
| Ribbed channel | 50-200 | Friction factor, Xr/h | Webb et al. 1971 |
| Urban canyon (side) | H/W 0.3-0.8 | Flow regime | Oke 1988 |
| Urban canyon (top-down) | 100-200 | Street-level wind | Qualitative |
| Building downwash | 100-200 | Cp distribution | Hunt 1984 |
| Ahmed body | Re=1000, slant 0-40 deg | Cd trend | Ahmed et al. 1984 |

## Key Features

- **MRT collision operator** (default) with independently tuned moment relaxation rates for stability up to Re~1000+. BGK fallback for comparison.
- **Bouzidi interpolated bounce-back** (2001) for smooth curved boundaries -- reduces stair-step Cd bias vs standard on-grid bounce-back. Supports both circles (cylinder) and arbitrary polygons (airfoil, Ahmed body).
- **Flat 1D memory layout** (std::vector) for cache-optimized access
- **OpenMP parallel** collision and streaming (collapse(2))
- **Momentum exchange** force extraction for Cd/Cl coefficients
- **Direct JSON output** -- per-frame velocity fields + append-only force history. No VTK bloat. Optional `--vtk` flag for legacy Paraview export.
- **Eight fluid analysis cases**: cylinder, lid-driven cavity, NACA airfoil, backward-facing step, ribbed channel, urban canyon (side + top-down views), building downwash, Ahmed body.
- **Polygon obstacle support** via point-in-polygon -- any closed 2D shape.
- **Production-grade**: Google Test suite (13 tests), GitHub Actions CI on ubuntu + macos.

## Interactive Website

The `docs/` directory contains an 8+ page portfolio website:

- **Project > Home** (index.html) -- Why build a custom LBM solver vs SU2/OpenFOAM, case table of contents
- **Simulation > [Case]** (cylinder.html, cavity.html, step.html, etc.) -- Per-case dedicated pages with field viewer (comparison slider), validation tables, force plots, discussion
- **Reference** (theory.html, implementation.html) -- LBM theory with KaTeX, code architecture with source blocks

Each case page has a draggable slider to compare velocity contours against streamlines, Re/AoA selector, and validation stats.

## Architecture

```
src/
  lbm_types.hpp     D2Q9 constants, MRT params, BounceBackGeometry, equilibrium
  lbm.hpp           Core solver: MRT collide, stream, Bouzidi BB, BCs, JSON output
  geometry.hpp      NACA 4-digit coords, polygon ops, point-in-polygon
  main.cpp          Cylinder flow
  cavity.cpp        Lid-driven cavity
  airfoil.cpp       NACA airfoil analysis
  step.cpp          Backward-facing step
  ribs.cpp          Ribbed channel flow
  urban_canyon.cpp  Urban canyon (--mode side|topdown)
  downwash.cpp      Building downwash
  ahmed.cpp         Ahmed body (2D slice)
  lbm_test.cpp      Google Test unit tests

scripts/
  postprocess.py    JSON -> PNG with --split, --cmap, --strouhal
  plot_airfoil.py   Airfoil validation plots
  run_*.sh          Batch sweep scripts

docs/
  index.html, theory.html, implementation.html, results.html
  css/style.css              CFD Jet theme
  assets/js/slider.js        Comparison slider
  assets/images/             Contour + streamline renders
```

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| A | Grid resolution 2x all cases | ⏳ In progress |
| B | Polygon Bouzidi interpolated bounce-back | ⏳ Not started |
| C | Urban canyon side-view + Oke 1988 regimes | ⏳ Not started |
| D | Ribbed channel friction factor + docs | ⏳ Not started |
| E | Postprocessor updates | ⏳ Not started |
| F | Run all sweeps + generate images | ⏳ Not started |
| G | Website per-case dedicated pages | ⏳ Not started |

Stretch goals: Smagorinsky LES, D3Q19 extension.

## License

MIT
