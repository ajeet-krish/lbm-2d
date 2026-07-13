# LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

A cache-optimized D2Q9 Lattice Boltzmann Method solver for 2D flow.
Features a **Multi-Relaxation Time (MRT) collision operator** (default,
BGK fallback), Bouzidi interpolated bounce-back for curved boundaries,
Smagorinsky LES turbulence model, and direct JSON output pipeline.
Validated against Blasius (flat plate), Williamson 1988 (Strouhal),
Tritton 1959 (drag), Ghia 1982 (lid-driven cavity), Armaly 1983
(backward-facing step), Oke 1988 (urban canyon), ERCOFTAC 043
(square cylinder), ISO 5167 (orifice plates),
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
./build/LBM_OrificePlate 100 3p          # Orifice plate (3 staggered plates)
./build/LBM_UrbanCanyon --mode side --ar 0.5  # Side-view canyon (3 buildings)
./build/LBM_UrbanCanyon --mode topdown 100    # Top-down street network (3 buildings)
./build/LBM_Downwash 100                 # Building downwash (tall + low-rise)

# New physics cases
./build/LBM_PeriodicHills 100            # Periodic hills (canonical LES benchmark)
./build/LBM_CylinderNearWall 100 20      # Cylinder near wall (ground effect, gap=20)
./build/LBM_SideBySide 100 3             # Side-by-side cylinders, transverse (S/D=3)
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
| Cylinder near wall (gap=10) | 800x300 | 100 | 0.590 | 0.1 | D=60 | Circle R=30 + wall | MRT | Bouzidi |
| Cylinder near wall (gap=20) | 800x300 | 100 | 0.590 | 0.1 | D=60 | Circle R=30 + wall | MRT | Bouzidi |
| Cylinder near wall (gap=40) | 800x300 | 100 | 0.590 | 0.1 | D=60 | Circle R=30 + wall | MRT | Bouzidi |
| Side-by-side S/D=2 (transverse) | 800x300 | 100 | 0.620 | 0.1 | D=40 | 2 circles, S=80 in y | MRT | Bouzidi |
| Side-by-side S/D=3 (transverse) | 800x300 | 100 | 0.620 | 0.1 | D=40 | 2 circles, S=120 in y | MRT | Bouzidi |
| Side-by-side S/D=5 (transverse) | 800x300 | 100 | 0.620 | 0.1 | D=40 | 2 circles, S=200 in y | MRT | Bouzidi |
| Rotating cylinder w=0.5 | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30, rotating | MRT | Bouzidi+Ladd |
| Rotating cylinder w=1.0 | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30, rotating | MRT | Bouzidi+Ladd |
| Rotating cylinder w=2.0 | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30, rotating | MRT | Bouzidi+Ladd |

### Internal Flows

| Case | Grid | Re | tau | u_ref | L_ref | Geometry | Collision | BC |
|------|------|-----|-----|-------|-------|----------|-----------|-----|
| Cavity | 800x300 | 100 | 0.680 | 0.1 | NY=300 | Square lid | MRT | Bounce-back |
| Cavity | 800x300 | 400 | 0.545 | 0.1 | NY=300 | Square lid | MRT | Bounce-back |
| Cavity | 800x300 | 1000 | 0.518 | 0.1 | NY=300 | Square lid | MRT+LES | Bounce-back |
| Step | 800x300 | 100 | 1.296 | 0.1 | Dh=398 | h_step=100 | MRT | Bounce-back |
| Step | 800x300 | 200 | 0.898 | 0.1 | Dh=398 | h_step=100 | MRT | Bounce-back |
| Step | 800x300 | 400 | 0.699 | 0.1 | Dh=398 | h_step=100 | MRT | Bounce-back |
| Sports-ball | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30 + dimple bumps | MRT | Bouzidi |
| Sports-ball (dimpled) | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30 + 12 bumps | MRT | Bouzidi |
| Periodic hills | 900x400 | 100 | 0.544 | 0.1 | H=267, h=H/6 | Sinusoidal bottom (3 periods) | MRT | Periodic x |
| Periodic hills | 900x400 | 1000 | 0.504 | 0.1 | H=267, h=H/6 | Sinusoidal bottom (3 periods) | MRT+LES | Periodic x |
| Periodic hills | 900x400 | 2800 | 0.502 | 0.1 | H=267, h=H/6 | Sinusoidal bottom (3 periods) | MRT+LES | Periodic x |

### Orifice Plate

| Case | Grid | Re | tau | u_inf | L_ref | Hole width | Geometry | Collision |
|------|------|-----|-----|-------|-------|-----------|----------|-----------|
| Orifice 1p1h | 800x300 | 100 | 0.725 | 0.025 | H=300 | 37 | 1 plate, 1 hole | MRT+LES |
| Orifice 1p3h | 800x300 | 100 | 0.725 | 0.025 | H=300 | 37 | 1 plate, 3 holes | MRT+LES |
| Orifice 2p | 800x300 | 100 | 0.725 | 0.025 | H=300 | 37 | 2 plates, staggered | MRT+LES |
| Orifice 3p | 800x300 | 100 | 0.725 | 0.025 | H=300 | 37 | 3 plates, staggered | MRT+LES |

### Urban / Building

| Case | Grid | Re | tau | u_ref | L_ref | Geometry | Collision |
|------|------|-----|-----|-------|-------|----------|-----------|
| Urban side AR=0.3 | 900x400 | 100 | 0.680 | 0.1 | H=100 | 2 bldgs, W=333 | MRT |
| Urban side AR=0.5 | 900x400 | 100 | 0.680 | 0.1 | H=100 | 2 bldgs, W=200 | MRT |
| Urban side AR=0.8 | 900x400 | 100 | 0.680 | 0.1 | H=100 | 2 bldgs, W=125 | MRT |
| Urban side AR=0.6 | 900x400 | 100 | 0.680 | 0.1 | H=100 | 3 bldgs, W=167 | MRT |
| Urban topdown (vertical) | 900x400 | 100 | 0.590 | 0.1 | w=80 | 3 tall bldgs, street net | MRT |
| Urban topdown (horizontal) | 900x400 | 100 | 0.590 | 0.1 | w=80 | 3 long bldgs, funnel | MRT |
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
| Orifice plate | 100 | Loss coeff K | ISO 5167, Idelchik 2006 |
| Periodic hills | 100-2800 | Reattachment, U-profile | Moser/Kim/Moin 1993 |
| Cylinder near wall | 100 | Cl vs gap | Ground effect literature |
| Side-by-side cylinders | 100 | Cd, Cl vs S/D | Zdravkovich 1977, Alam & Zhou 2007 |
| Rotating cylinder | 100 | Cl vs omega | Kutta-Joukowski theorem |
| Urban canyon (side) | H/W 0.3-0.8 | Flow regime | Oke 1988 |
| Building downwash | 100 | Cp distribution | Hunt 1984 |

## Key Features

- **MRT collision operator** (default) with independently tuned moment relaxation rates for stability up to Re~1000+. BGK fallback for comparison.
- **Smagorinsky LES** subgrid-scale turbulence model with automatic activation when tau < 0.55 (high Re).
- **Bouzidi interpolated bounce-back** (2001) for smooth curved boundaries -- reduces stair-step Cd bias vs standard on-grid bounce-back. Supports both circles (cylinder) and arbitrary polygons (flat plate, square cylinder).
- **Flat 1D memory layout** (std::vector) for cache-optimized access
- **OpenMP parallel** collision and streaming (collapse(2))
- **Momentum exchange** force extraction for Cd/Cl coefficients
- **Direct JSON output** -- per-frame velocity, pressure, vorticity fields + append-only force history. Optional `--vtk` flag for legacy Paraview export.
- **15 simulation cases**: flat plate, cylinder, square cylinder, lid-driven cavity, backward-facing step, sports-ball (surface roughness), orifice plate, urban canyon (side + topdown vertical/horizontal), building downwash, periodic hills, cylinder near wall, side-by-side cylinders, rotating cylinder.
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
  orifice_plate.cpp    Orifice plate (single + multi-stage, staggered)
  sports_ball.cpp      Sports-ball surface roughness (dimpled cylinder, drag reduction)
  urban_canyon.cpp     Urban canyon (side 2b/3b + topdown vertical/horizontal)
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
  sports_ball.html     Sports-ball surface roughness (dimples, drag reduction)
  orifice_plate.html   Orifice plate (single + multi-stage)
  urban.html           Urban canyon (side + topdown vertical/horizontal + downwash)
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
| Ribs/sports-ball | 100 | Cd vs roughness | -- | Dimple drag reduction |
| Orifice plate | 100 | Fx 0.9-63 | -- | K increases with plates |
| Periodic hills | 100-2800 | -- | -- | LES benchmark |
| Cylinder near wall | 100 | 2.6-2.8 | +0.40 to +1.42 | Ground effect (lift vs gap) |
| Side-by-side | 100 | 2.6-2.8 | ~0 (amp 0.6-0.7) | Interference study |
| Rotating cylinder | 100 | -- | -- | Magnus effect (Ladd) |
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

### Pending Fixes (Phase 4)

| Fix | Problem | Solution | Priority |
|-----|---------|----------|----------|
| Ladd moving boundary | Rotating cylinder has no tangential velocity (Cl~0) | Implement Ladd (1994) bounce-back with wall velocity in `lbm.hpp` | **Completed** |
| Cylinder near wall | Cylinder too low (touching ground, no under-flow) | Raise cylinder to gaps 10/20/40 cells | **Completed** |
| Side-by-side geometry | Was tandem (same y, offset x) | Rebuilt as transverse (same x, offset y), D=40 to fit domain | **Completed** |
| Orifice single-hole jet | 1p1h/2p/3p diverged (jet Mach too high) | Lower u_inflow to 0.025 + enable LES | **Completed** |
| Cylinder Re=1000 | Diverged at step 16k before auto-LES was available | Add auto-LES guard to `main.cpp` and rerun | High |

## License

MIT
