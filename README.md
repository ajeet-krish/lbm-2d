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

# PINN surrogate training (requires PyTorch)
cd pinn && pip3 install -r requirements.txt
python3 train.py                          # Cylinder Re=100 (original)
python3 train_cavity.py                   # Cavity multi-Re (Re=100 + Re=400)
python3 train_cavity.py --single-re 100   # Cavity single Re
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
| Sports-ball | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30 + 16 dimple bumps | MRT | Bouzidi |
| Sports-ball (dimpled) | 800x300 | 100 | 0.680 | 0.1 | D=60 | Circle R=30 + 16 bumps | MRT | Bouzidi |
| Periodic hills | 800x300 | 100 | 1.10 | 0.1 | H=200, h=H/6 | Sinusoidal bottom (1 period, L=NX) | MRT | Periodic x |
| Periodic hills | 800x300 | 1000 | 0.56 | 0.1 | H=200, h=H/6 | Sinusoidal bottom (1 period, L=NX) | MRT+LES | Periodic x |
| Periodic hills | 800x300 | 2800 | 0.52 | 0.1 | H=200, h=H/6 | Sinusoidal bottom (1 period, L=NX) | MRT+LES | Periodic x |

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
| Sports-ball roughness | 100 | Cd smooth vs dimpled | Golf-ball drag analogy |
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
- **14 simulation cases**: flat plate, cylinder, square cylinder, lid-driven cavity, backward-facing step, orifice plate, urban canyon (side + topdown vertical/horizontal), building downwash, periodic hills, cylinder near wall, side-by-side cylinders, rotating cylinder.
  - **PINN surrogate**: Fourier-feature parametric PINN (593K params) trained on cavity Re=100+400, 3-panel comparison (LBM/PINN/Error) on website, interactive Re-sweep slider.
  - **Interactive Flow Viewer**: Per-case canvas engine (`docs/assets/js/flow-viewer.js`) streams compact float16 binary frame data (velocity magnitude + streamlines) with Play/Pause + scrubber. Live PINN tab runs the surrogate in-browser via ONNX Runtime Web.
- **Polygon obstacle support** via point-in-polygon -- any closed 2D shape.
- **Production-grade**: Google Test suite (12 tests), GitHub Actions CI on ubuntu + macos.

## Physics-Informed Neural Network (PINN) Surrogate Suite

A mesh-free parametric surrogate suite in `pinn/`. Trains PyTorch PINNs on
C++ LBM output as hybrid data-physics surrogates. Deploys via ONNX Runtime Web
for real-time interactive demos. Mirrors the SciML R&D pipeline at NASA,
Rolls-Royce, and F1 teams.

**Parametric architecture:** Pass physical parameters (Re, geometry dimensions)
directly into the network alongside spatial coordinates:
```
[x, y, Re, hole_w, ...] --> PINN --> [u, v, p]
```
A recruiter drags a slider -- the flow field updates instantly, no retraining.

**Spectral-bias fix:** Standard tanh MLPs suffer from spectral bias and
under-represent high-frequency boundary layers. The cavity surrogate uses a
Fourier feature embedding (frozen random sinusoidal projection of spatial
coords) before the MLP to lift inputs into a high-frequency space, breaking
this limitation. See `pinn/models/pinn.py` (`FourierFeatureLayer`).

**Cavity results (v3, Fourier features, 593K params):**

| Re | L2 u | L2 v | u_max ratio | Status |
|----|-------|-------|-------------|--------|
| 100 | 23.7% | 29.3% | 1.24 | Trained |
| 400 | 24.4% | 30.0% | 1.10 | Trained |
| 200 | -- | -- | -- | Interpolated (not in training data) |

u_max ratio improved from 3.50 (v2, plain tanh) to 1.24 (v3, Fourier).
Velocity field error dropped 30x. Parametric Re-slider demo on `cavity.html`.

**Implementation order:**

| Phase | Case | Parametric Axis | Data Status | Portfolio Demo |
|-------|------|----------------|-------------|----------------|
| 6.3 | Lid-driven cavity | Re (100-400) | Exists (51 frames, 128x128) | Re slider -> vortex center shift |
| 6.4 | Backward-facing step | Re (100-400) | Re-run Re=100 needed (no p/omega) | Re slider -> reattachment length |
| 6.5 | Orifice plate | hole_w, n_plates | New Re+geometry sweeps needed | Diameter slider -> loss coeff K |
| 6.8 | Time-Parametric PINN (NEW) | t + Re | LBM time-series at Re=100+400 | Watch vortex roll-up at any timestep |

#### Phase 6.8: Time-Parametric PINN (Spatio-Temporal Surrogate)

**Goal:** Extend the parametric PINN to a spatio-temporal surrogate that learns the
full transient evolution of the flow, not just the steady state. Input becomes
`(x, y, Re_n, t_n)` where `t_n = frame_index / (n_frames - 1)` is the normalized
simulation time. The network predicts `(u, v, p)` at any point in space AND time,
enabling true ML-powered animations: a recruiter drags a time slider and watches
the vortex roll-up, shedding, and approach to steady state -- all from a single
neural network, no CFD re-run.

**Architecture:** Fourier features on spatial coords (x, y) only -> 512-dim, then
concatenate `Re_n` and `t_n` -> 514-dim MLP input. MLP: 256 hidden, 8 layers, tanh
(~600K params). Output: `(u, v, p)`.

**Unsteady PDE residual (vs steady NS in 6.3):** Add the material time derivative:
```
du/dt + u·du/dx + v·du/dy = -dp/dx + nu*(d2u/dx2 + d2u/dy2)
dv/dt + u·dv/dx + v·dv/dy = -dp/dy + nu*(d2v/dx2 + d2v/dy2)
du/dx + dv/dy = 0  (continuity)
```
`du/dt` and `dv/dt` are computed via torch.autograd from the `t` input.

**Training data:** 51-frame LBM sequences at Re=100 and Re=400 (importance-sampled
sensors, 3000/frame). Hybrid loss = w_pde*unsteady_NS + w_data*data(u,v,p) + w_bc*BC.

**Portfolio value:** The truly compelling ML story. A physics-informed network that
learns temporal dynamics from the LBM time-series, then predicts the entire
spatio-temporal flow field in real-time browser inference -- the solver generates
baseline data once, the PINN provides a deployable, interactive surrogate.

**Status:** Pending (architecture + training script planned; cavity first).

**Web integration:** Each case page gets two separate viewer sections -- "LBM
Evolution" (C++ solver frames) and "PINN Prediction" (surrogate) -- so the solver
and ML results are shown side by side for direct comparison.

**Key features:**
- Trains on Apple Silicon via PyTorch MPS backend (no CUDA)
- Hybrid loss = data + PDE residual (steady incompressible NS) + BC loss
- Fourier feature embedding to mitigate tanh spectral bias
- 3-panel comparison (LBM / PINN / error delta) on website
- Zero changes to the existing C++ solver

See `pinn/README.md` for setup, architecture, and the full phased plan.

## Interactive Website

The `docs/` directory contains a 12+ page portfolio website:

- **Project > Home** (index.html) -- Why build a custom LBM solver vs SU2/OpenFOAM, case table of contents
- **Simulation > [Case]** -- Per-case dedicated pages with interactive flow viewers, validation tables, force plots, discussion
- **Reference** (theory.html, implementation.html) -- LBM theory with KaTeX, code architecture with source blocks

Each case page has two interactive viewer sections:

1. **LBM Evolution** -- Animated canvas streaming the C++ solver's velocity field
   from rest to steady state (Play/Pause + scrubber). Velocity magnitude contours
   with overlaid streamlines.
2. **PINN Prediction** -- The parametric surrogate (precomputed sweep or live ONNX
   Runtime Web inference). Reynolds number slider drives the network in real-time.
   Time-parametric PINN (Phase 6.8) will add a time scrubber here so the surrogate
   animates the flow evolution like the LBM section.

Binary frame data is exported by `pinn/export_web_data.py` to `docs/assets/data/{case}/`
as float16 `.bin` files (gzipped for delivery).

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
  cavity.html          Lid-driven cavity + PINN parametric Re-sweep
  step.html            Backward-facing step
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
| Sports-ball (smooth) | 100 | 1.703 | ~0 | Baseline; wider wake |
| Sports-ball (dimpled) | 100 | 1.902 | ~0 | Dimples raise drag at Re=100 (low-Re regime) |
| Orifice plate | 100 | Fx 0.9-63 | -- | K increases with plates |
| Periodic hills | 100-2800 | -- | -- | LES benchmark (re-run pending after L=NX fix) |
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
| 5 | Website updates for new features | In progress (interactive viewers on all cases) |
| 6 | Physics-Informed Neural Network (PINN) surrogate suite | In progress (cavity done, step/orifice pending, temporal PINN planned) |
| 7 | Time-parametric PINN training (Phase 6.8) | Pending |

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
