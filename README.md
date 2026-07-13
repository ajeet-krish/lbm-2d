# LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

A cache-optimized D2Q9 Lattice Boltzmann Method solver for 2D flow.
Features a **Multi-Relaxation Time (MRT) collision operator** (default,
BGK fallback), Bouzidi interpolated bounce-back for curved boundaries,
Smagorinsky LES turbulence model, and direct JSON output pipeline.
Validated against Blasius (flat plate), Williamson 1988 (Strouhal),
Tritton 1959 (drag), Ghia 1982 (lid-driven cavity), Armaly 1983
(backward-facing step), Oke 1988 (urban canyon), ERCOFTAC 043
(square cylinder), Bernoulli (converging-diverging nozzle),
Moser/Kim/Moin 1993 (periodic hills), and Kutta-Joukowski (Magnus effect).

Built as an aerospace/defense portfolio piece demonstrating HPC competency
(C++20, OpenMP), CFD fundamentals (MRT-LBM, Bouzidi, Smagorinsky LES,
momentum exchange), and engineering communication skills (interactive
web results with per-case dedicated pages and comparison sliders).

## Quick Start

```bash
cmake -B build && cmake --build build -j$(sysctl -n hw.ncpu)

# Primary validation case
./build/LBM_FlatPlate 1000 0             # Flat plate at Re=1000, AoA=0
./build/LBM_FlatPlate 1000 10            # Flat plate at Re=1000, AoA=10 deg

# Fluid analysis cases
./build/LBM_Engine 100                   # Cylinder wake at Re=100 (curved boundary demo)
./build/LBM_SquareCylinder 200           # Square cylinder at Re=200 (ERCOFTAC 043)
./build/LBM_Cavity 100                   # Lid-driven cavity at Re=100
./build/LBM_Step 100                     # Backward-facing step at Re=100
./build/LBM_Ribs 100                     # Ribbed channel at Re=100
./build/LBM_Nozzle 100 0.25             # CD nozzle, area ratio=0.25
./build/LBM_UrbanCanyon --mode side --ar 0.5  # Side-view canyon (3 buildings)
./build/LBM_UrbanCanyon --mode topdown 100    # Top-down street network (3 buildings)
./build/LBM_Downwash 100                 # Building downwash (tall + low-rise)

# New physics cases
./build/LBM_PeriodicHills 100            # Periodic hills (canonical LES benchmark)
./build/LBM_CylinderNearWall 100 4       # Cylinder near wall (ground effect, gap=4)
./build/LBM_SideBySide 100 3             # Side-by-side cylinders (S/D=3)
./build/LBM_RotatingCylinder 100 1.0     # Rotating cylinder (Magnus, omega=1.0)

# Run tests
./build/LBM_Tests

# Post-process (separate contour + streamline PNGs)
python3 scripts/postprocess.py output/flatplate/re1000_aoa0 --split
python3 scripts/postprocess.py output/cylinder/re100 --split --vorticity

# Preview website
python3 -m http.server -d docs 8765
open http://localhost:8765
```

## Simulation Parameters

### External Aerodynamics

| Case | Grid | Re | tau | u_inf | L_ref | Geometry | Collision | BC |
|------|------|-----|-----|-------|-------|----------|-----------|-----|
| Cylinder | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30 | MRT | Bouzidi |
| Cylinder | 800x300 | 200 | 0.590 | 0.1 | D=60 | Circle R=30 | MRT | Bouzidi |
| Cylinder | 800x300 | 1000 | 0.518 | 0.1 | D=60 | Circle R=30 | MRT+LES | Bouzidi |
| Square cylinder | 800x300 | 200 | 0.545 | 0.1 | s=30 | Square 30x30 | MRT | Polygon |
| Flat plate AoA=0 | 800x300 | 1000 | 0.547 | 0.1 | c=155 | Rectangle 155x2 | MRT | Polygon |
| Flat plate AoA=10 | 800x300 | 1000 | 0.547 | 0.1 | c=155 | Rectangle 155x2 | MRT | Polygon |
| Flat plate Re=500 | 800x300 | 500 | 0.593 | 0.1 | c=155 | Rectangle 155x2 | MRT | Polygon |
| Flat plate Re=2000 | 800x300 | 2000 | 0.523 | 0.1 | c=155 | Rectangle 155x2 | MRT+LES | Polygon |
| Cylinder near wall (gap=2) | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30 + wall | MRT | Bouzidi |
| Cylinder near wall (gap=4) | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30 + wall | MRT | Bouzidi |
| Cylinder near wall (gap=8) | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30 + wall | MRT | Bouzidi |
| Side-by-side S/D=2 | 800x300 | 100 | 0.680 | 0.1 | D=60 | 2 circles, S=120 | MRT | Bouzidi |
| Side-by-side S/D=3 | 800x300 | 100 | 0.680 | 0.1 | D=60 | 2 circles, S=180 | MRT | Bouzidi |
| Side-by-side S/D=5 | 800x300 | 100 | 0.680 | 0.1 | D=60 | 2 circles, S=300 | MRT | Bouzidi |
| Rotating cylinder w=0.5 | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30, rotating | MRT | Bouzidi+spin |
| Rotating cylinder w=1.0 | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30, rotating | MRT | Bouzidi+spin |
| Rotating cylinder w=2.0 | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30, rotating | MRT | Bouzidi+spin |

### Internal Flows

| Case | Grid | Re | tau | u_ref | L_ref | Geometry | Collision | BC |
|------|------|-----|-----|-------|-------|----------|-----------|-----|
| Cavity | 800x300 | 100 | 0.680 | 0.1 | NY=300 | Square lid | MRT | Bounce-back |
| Cavity | 800x300 | 400 | 0.545 | 0.1 | NY=300 | Square lid | MRT | Bounce-back |
| Cavity | 800x300 | 1000 | 0.518 | 0.1 | NY=300 | Square lid | MRT+LES | Bounce-back |
| Step | 800x300 | 100 | 1.296 | 0.1 | Dh=398 | h_step=100 | MRT | Bounce-back |
| Step | 800x300 | 200 | 0.898 | 0.1 | Dh=398 | h_step=100 | MRT | Bounce-back |
| Step | 800x300 | 400 | 0.699 | 0.1 | Dh=398 | h_step=100 | MRT | Bounce-back |
| Ribs | 400x200 | 50 | 2.100 | 0.1 | Dh=400 | h_rib=10, p=100 | MRT | Periodic |
| Ribs | 400x200 | 100 | 1.300 | 0.1 | Dh=400 | h_rib=10, p=100 | MRT | Periodic |
| Ribs | 400x200 | 200 | 0.900 | 0.1 | Dh=400 | h_rib=10, p=100 | MRT | Periodic |
| Periodic hills | 900x400 | 100 | 0.544 | 0.1 | H=267 | Sinusoidal bottom | MRT | Periodic x |
| Periodic hills | 900x400 | 1000 | 0.504 | 0.1 | H=267 | Sinusoidal bottom | MRT+LES | Periodic x |
| Periodic hills | 900x400 | 2800 | 0.502 | 0.1 | H=267 | Sinusoidal bottom | MRT+LES | Periodic x |

### Nozzle

| Case | Grid | Re | tau | u_inf | L_ref | AR | Geometry | Collision |
|------|------|-----|-----|-------|-------|----|----------|-----------|
| Nozzle AR=0.25 | 800x300 | 100 | 0.722 | 0.1 | D_th=74 | 0.25 | Cosine wall | MRT |
| Nozzle AR=0.25 | 800x300 | 500 | 0.544 | 0.1 | D_th=74 | 0.25 | Cosine wall | MRT |
| Nozzle AR=0.25 | 800x300 | 1000 | 0.522 | 0.1 | D_th=74 | 0.25 | Cosine wall | MRT+LES |
| Nozzle AR=0.50 | 800x300 | 100 | 0.722 | 0.1 | D_th=74 | 0.50 | Cosine wall | MRT |
| Nozzle AR=0.50 | 800x300 | 500 | 0.544 | 0.1 | D_th=74 | 0.50 | Cosine wall | MRT |
| Nozzle AR=0.50 | 800x300 | 1000 | 0.522 | 0.1 | D_th=74 | 0.50 | Cosine wall | MRT+LES |

### Urban / Building

| Case | Grid | Re | tau | u_ref | L_ref | Geometry | Collision |
|------|------|-----|-----|-------|-------|----------|-----------|
| Urban side AR=0.3 | 900x400 | 100 | 0.680 | 0.1 | H=100 | 3 bldgs, W=333 | MRT |
| Urban side AR=0.5 | 900x400 | 100 | 0.680 | 0.1 | H=100 | 3 bldgs, W=200 | MRT |
| Urban side AR=0.8 | 900x400 | 100 | 0.680 | 0.1 | H=100 | 3 bldgs, W=125 | MRT |
| Urban topdown | 900x400 | 100 | 0.590 | 0.1 | w=60 | 3 bldgs, street net | MRT |
| Downwash | 800x300 | 100 | 0.740 | 0.1 | H=120 | Tall(120)+low(45) | MRT |

## Validation Coverage

| Case | Re Range | Key Metric | Literature |
|------|----------|-----------|------------|
| **Flat plate boundary layer** | 500-2000 | Cf, Cd, Cl vs AoA | Blasius 1908, thin-airfoil theory |
| Cylinder wake | 100-200 | St, Cd | Williamson 1988, Tritton 1959 |
| Square cylinder | 200 | Cd, St | Lyn et al. 1995 (ERCOFTAC 043) |
| Lid-driven cavity | 100-1000 | u-profile | Ghia, Ghia & Shin 1982 |
| Backward-facing step | 100-400 | Xr/H | Armaly et al. 1983 |
| Ribbed channel | 50-200 | Friction factor | Webb et al. 1971 |
| Converging-diverging nozzle | 100-1000 | Cp, v(x) | Bernoulli equation |
| Periodic hills | 100-2800 | Reattachment, U-profile | Moser/Kim/Moin 1993 |
| Cylinder near wall | 100 | Cd vs gap | Ground effect literature |
| Side-by-side cylinders | 100 | Cd vs S/D | Zdravkovich 1977 |
| Rotating cylinder | 100 | Cl vs omega | Kutta-Joukowski theorem |
| Urban canyon (side) | H/W 0.3-0.8 | Flow regime | Oke 1988 |
| Building downwash | 100 | Cp distribution | Hunt 1984 |

## Key Features

- **MRT collision operator** (default) with independently tuned moment relaxation rates for stability up to Re~1000+. BGK fallback for comparison.
- **Smagorinsky LES** subgrid-scale turbulence model with automatic activation when tau < 0.55 (high Re).
- **Bouzidi interpolated bounce-back** (2001) for smooth curved boundaries -- reduces stair-step Cd bias vs standard on-grid bounce-back. Supports both circles (cylinder) and arbitrary polygons (flat plate, nozzle).
- **Flat 1D memory layout** (std::vector) for cache-optimized access
- **OpenMP parallel** collision and streaming (collapse(2))
- **Momentum exchange** force extraction for Cd/Cl coefficients
- **Direct JSON output** -- per-frame velocity, pressure, vorticity fields + append-only force history. Optional `--vtk` flag for legacy Paraview export.
- **15 simulation cases**: flat plate, cylinder, square cylinder, lid-driven cavity, backward-facing step, ribbed channel, converging-diverging nozzle, urban canyon (side + topdown), building downwash, periodic hills, cylinder near wall, side-by-side cylinders, rotating cylinder.
- **Polygon obstacle support** via point-in-polygon -- any closed 2D shape.
- **Production-grade**: Google Test suite (12 tests), GitHub Actions CI on ubuntu + macos.

## Interactive Website

The `docs/` directory contains a 12+ page portfolio website:

- **Project > Home** (index.html) -- Why build a custom LBM solver vs SU2/OpenFOAM, case table of contents
- **Simulation > [Case]** -- Per-case dedicated pages with field viewer (comparison slider), validation tables, force plots, discussion
- **Reference** (theory.html, implementation.html) -- LBM theory with KaTeX, code architecture with source blocks

Each case page has a draggable slider to compare velocity contours against streamlines, Re/AoA selector, and validation stats.

## Architecture

```
src/
  lbm_types.hpp        D2Q9 constants, MRT params, BounceBackGeometry, equilibrium
  lbm.hpp              Core solver: MRT collide, LES, stream, Bouzidi BB, BCs, JSON output
  geometry.hpp         NACA 4-digit coords, polygon ops, point-in-polygon
  main.cpp             Cylinder flow (auto-LES for tau < 0.55)
  flat_plate.cpp       Flat plate boundary layer (PRIMARY validation case)
  cavity.cpp           Lid-driven cavity
  step.cpp             Backward-facing step
  square_cylinder.cpp  Square cylinder (ERCOFTAC 043)
  nozzle.cpp           Converging-diverging nozzle (Bernoulli validation)
  ribs.cpp             Ribbed channel flow
  urban_canyon.cpp     Urban canyon (--mode side|topdown, 3 buildings)
  downwash.cpp         Building downwash (scaled buildings)
  periodic_hills.cpp   Periodic hills (canonical LES benchmark)
  cylinder_near_wall.cpp  Cylinder near wall (ground effect)
  side_by_side_cylinders.cpp  Side-by-side cylinders (interference)
  rotating_cylinder.cpp  Rotating cylinder (Magnus effect)
  amr.hpp              AMRBlock, AMRGrid, refinement, regridding
  lbm_test.cpp         Google Test unit tests

scripts/
  postprocess.py       JSON -> PNG with --split, --cmap, --strouhal, --vorticity
  run_*.sh             Batch sweep scripts

docs/
  flat_plate.html      PRIMARY validation case (Blasius, drag polar)
  cylinder.html        Cylinder wake (comparison slider)
  square_cylinder.html ERCOFTAC 043 (sharp-edge separation)
  cavity.html          Lid-driven cavity
  step.html            Backward-facing step
  ribs.html            Ribbed channel
  nozzle.html          CD nozzle (Bernoulli validation)
  urban.html           Urban canyon (side + topdown + downwash)
  periodic_hills.html  Periodic hills (LES benchmark)
  cylinder_near_wall.html  Cylinder near wall (ground effect)
  side_by_side.html    Side-by-side cylinders (interference)
  rotating_cylinder.html  Rotating cylinder (Magnus effect)
  theory.html          LBM theory with KaTeX
  implementation.html  Code architecture
  index.html           Home / landing page
  css/style.css        CFD Jet theme (dark, cyan/orange accents)
  assets/js/slider.js  Comparison slider
  assets/images/       Contour + streamline renders per case
```

## Simulation Results Summary

| Case | Re | Cd | Cl | Status |
|------|-----|-----|-----|--------|
| Flat plate AoA=0 | 1000 | ~0.026 | 0 | Validated vs Blasius |
| Flat plate AoA=10 | 1000 | ~0.05 | ~1.1 | Separation expected |
| Cylinder | 100 | 1.774 | ~0 | Validated |
| Cylinder | 200 | 1.495 | ~0 | Validated |
| Square cylinder | 200 | 1.157 | 0.47 | Validated vs ERCOFTAC |
| Cavity | 100-1000 | -- | -- | Validated vs Ghia |
| Step | 100-400 | -- | -- | Validated vs Armaly |
| Ribs | 50-200 | 0.26-0.64 | 0 | Validated |
| Nozzle (AR=0.25) | 100-1000 | -- | -- | Bernoulli validation |
| Periodic hills | 100-2800 | -- | -- | LES benchmark |
| Cylinder near wall | 100 | -- | -- | Ground effect study |
| Side-by-side | 100 | -- | -- | Interference study |
| Rotating cylinder | 100 | -- | -- | Magnus effect |
| Urban canyon | 100 | -- | -- | Oke 1988 regimes |
| Downwash | 100 | -- | -- | Hunt 1984 |

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Solver Improvement Plan (correctness + perf + cleanup) | Completed |
| 1 | Smagorinsky LES turbulence model | Completed |
| 2 | Block-structured AMR (adaptive mesh refinement) | In progress |
| 3 | Vorticity output + postprocessor | Completed |
| 4 | Full simulation re-runs + new cases | In progress |
| 5 | Website updates for new features | Pending |

## License

MIT
