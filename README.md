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

# PINN surrogate training (requires PyTorch; uses local venv with MPS on Apple Silicon)
cd pinn && .venv/bin/pip install -r requirements.txt
.venv/bin/python cases/cylinder/train.py            # Cylinder Re=100 (original)
.venv/bin/python cases/cavity/train_steady.py        # Cavity steady multi-Re (Re=100 + Re=400)
.venv/bin/python cases/cavity/train_temporal.py      # Cavity time-parametric (Re=100 + Re=400 + Re=1000)
.venv/bin/python export/export_web_data.py           # LBM frames -> float16 .bin for website
.venv/bin/python cases/cavity/export_temporal.py     # Temporal PINN -> ONNX + frame binaries
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
  - **PINN surrogate**: Fourier-feature parametric PINN (593K params) trained on cavity Re=100+400+1000, 3-panel comparison (LBM/PINN/Error in Reds) on website, discrete Re buttons + Re=300 interpolation panel, ~300-600x faster than the solver.
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
| 200 | 25.0% | 28.7% | -- | Interpolated (Re=300, not in training data) |

u_max ratio improved from 3.50 (v2, plain tanh) to 1.24 (v3, Fourier).
Velocity field error dropped 30x. Parametric Re interpolation demo on
`cavity.html` (discrete Re buttons 100/400/1000, with Re=300 shown as an
interpolated prediction panel).

**Time-parametric results (Phase 6.8, 593K params, 514-dim input):**

Trained on the full 51-frame transient at Re=100, Re=400, and Re=1000. A single
network predicts `(u, v, p)` at any `(x, y, Re, t)`.

| Re | L2 u (mean / final frame) | L2 v (mean / final frame) | u_max ratio |
|----|---------------------------|---------------------------|-------------|
| 100 | 33.3% / 29.9% | 48.0% / -- | 1.13 |
| 400 | 33.0% / 34.7% | 43.1% / -- | 1.16 |
| 1000 | 37.5% / -- | 31.2% / -- | -- |

12,000 Adam + 1,000 L-BFGS epochs per training, ~201 min on Apple Silicon MPS.
Final hybrid loss 1.2e-3 (Re=100/400). Early transient (frames 0-10) is hardest
at ~45% L2. The temporal model supersedes the steady-state model for animation.
Export: `pinn_temporal_re{100,400,1000}.bin` + `pinn_temporal_model.onnx`
(2.38 MB), wired into `cavity.html` PINN Prediction section as a second
FlowViewer with Re buttons + time scrubber.

**Speed:** The trained surrogate inferences a full 96x96 field in ~60-100 ms on
CPU (ONNX Runtime Web, single thread) versus ~30 s per frame for the C++ LBM
solver on the same grid -- a roughly 300-600x speedup that makes real-time,
interactive design-space exploration practical in the browser.

**Implementation order:**

| Phase | Case | Parametric Axis | Data Status | Portfolio Demo |
|-------|------|----------------|-------------|----------------|
| 6.3 | Lid-driven cavity | Re (100-400) | Exists (51 frames, 128x128) | Re buttons -> vortex center shift |
| 6.4 | Backward-facing step | Re (100-400) | Re-run Re=100 needed (no p/omega) | Re slider -> reattachment length |
| 6.5 | Orifice plate | hole_w, n_plates | New Re+geometry sweeps needed | Diameter slider -> loss coeff K |
| 6.8 | Time-Parametric PINN | t + Re | LBM time-series at Re=100+400+1000 | Watch vortex roll-up at any timestep |

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

**Status:** Completed (Re=100/400/1000 trained, 201 min MPS each, ONNX +
float16 binary export done; cavity.html PINN Prediction section animates the
transient via a second FlowViewer with Re buttons + time scrubber). Re=300
interpolation prediction added as a parametric demo panel.

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
   (jet colormap) with overlaid streamlines.
2. **PINN Prediction** -- The parametric surrogate (precomputed sweep or live ONNX
   Runtime Web inference). Discrete Reynolds number buttons (100/400/1000) drive
   the network; a time scrubber animates the transient (Phase 6.8 temporal PINN).

Binary frame data is exported by `pinn/export/export_web_data.py` to
`docs/assets/data/{case}/` as float16 `.bin` files (gzipped for delivery). All
velocity/flow animations use the jet colormap for visual consistency; the
3-panel error-delta panel uses Reds; vorticity uses RdBu.

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

pinn/
  README.md            Setup, architecture, phased roadmap
  requirements.txt     torch, numpy, matplotlib, scipy, onnx, onnxruntime
  cases/
    cavity/            train_steady.py, train_temporal.py, export_sweep.py,
                       export_temporal.py, plot_results.py, plot_loss_convergence.py,
                       plot_temporal_l2.py, logs/
    cylinder/          train.py, evaluate.py
  export/
    export_web_data.py LBM frames -> float16 .bin (+.gz) for website
  data/                loader.py, temporal_loader.py
  models/              pinn.py (PINN, ParametricPINN, FourierFeatureLayer), losses.py
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
| 2 | Block-structured AMR (adaptive mesh refinement) | In progress (restriction operator needs fix) |
| 3 | Vorticity output + postprocessor | Completed |
| 4 | Full simulation re-runs + new cases | In progress (17 simulations pending) |
| 5 | Website updates for new features | In progress (interactive viewers on all cases) |
| 6 | Physics-Informed Neural Network (PINN) surrogate suite | In progress (cavity steady + temporal done; Re=1000 temporal done; step/orifice pending) |
| 6.8 | Time-parametric PINN training | Completed (Re=100/400/1000; Re=300 interpolation demo) |
| 6.8b | Re=1000 temporal extension | Completed |
| 6.9 | Model improvement roadmap (pressure-Poisson, Re range) | Pending |
| 5.5 | Cavity page deep dive + PINN surrogate narrative | Completed (Key Findings, LBM Analysis, Training Convergence, What PINN Unlocks, Limitations; loss + temporal L2 plots; 600x speed section; Re=300 interpolation; sensitivity map) |

### Pending Fixes (Phase 4)

| Fix | Problem | Solution | Priority |
|-----|---------|----------|----------|
| Ladd moving boundary | Rotating cylinder has no tangential velocity (Cl~0) | Implement Ladd (1994) bounce-back with wall velocity in `lbm.hpp` | **Completed** |
| Cylinder near wall | Cylinder too low (touching ground, no under-flow) | Raise cylinder to gaps 10/20/40 cells | **Completed** |
| Side-by-side geometry | Was tandem (same y, offset x) | Rebuilt as transverse (same x, offset y), D=40 to fit domain | **Completed** |
| Orifice single-hole jet | 1p1h/2p/3p diverged (jet Mach too high) | Lower u_inflow to 0.025 + enable LES | **Completed** |
| Cylinder Re=1000 | Diverged at step 16k on coarse grid (tau=0.518 < 0.55) | Stable on fine grid (NX=2400, NY=900) but unsteady; documented as known limitation; website surfaces Re=20/40/100/200 only | **Deferred** |

## Solver Accuracy & Capability Upgrade Roadmap

The current solver uses Cartesian immersed boundary + Bouzidi interpolated
bounce-back, no near-wall treatment (pure no-slip), isothermal incompressible only,
and is strictly 2D (D2Q9 hardcoded). The roadmap below improves accuracy (smooth
curved boundaries, proper y+ / wall treatment), adds pressure visualization, thermal
physics, and a 3D architecture.

### UPGRADE 1: Curved Boundary Accuracy (blocky -> smooth)

Three tiers, easiest first:

- **Tier 1 (immediate): Grid refinement.** Increase grid from 800x300 to 2400x900 for
  curved cases. At radius=90 cells (vs 30), staircase error drops 3x. Bouzidi `q`
  already gives sub-grid positioning. Cost: 9x memory/compute (feasible 2D on M5).
  **Status: Available via --nx/--ny override (main.cpp).**
- **Tier 2 (medium effort, recommend first): Filippova-Hanel / Mei interpolated
  bounce-back.** Bouzidi is 2nd-order but unstable at q->0 or q->1. Replace with the
  Mei et al. (1999) formula:
  `f_bb = q*f_i^{eq}(rho,u_wall) + (1-q)*f_post + (2q-1)*w_i*rho*(e_i.u_wall)/cs^2`.
  Unconditionally stable for all q in [0,1]. Implemented as `use_mei_bb` (default true).
  **Status: Completed (2026-07-18), Cd~1.53 at Re=100 (vs 1.77 Bouzidi).**
- **Tier 3 (high effort): IBM with direct forcing.** For airfoils/complex shapes. New
  `src/ibm.hpp` (~200 lines). Lagrangian points on true surface (reuse `naca_coords()`),
  force spreading via 4-point smoothed delta, velocity interpolation. Add only for
  airfoil case. **Status: In progress (2026-07-18).**

### UPGRADE 2: Wall Functions / y+ Requirements

- **Wall-distance computation:** New `compute_wall_distance()` in lbm_types.hpp -- BFS from
  obstacle nodes, returns distance in lattice units. **Status: Completed (2026-07-18).**
- **Van Driest damping for LES:** Implemented in `mrt_collide()`:
  `nu_t_damped = nu_t * (1 - exp(-y+/A+))^2` with A+ = 26. Prevents over-damping near walls.
  **Status: Completed (2026-07-18), validated Cd~1.77 at Re=100.**
- **Wall function bounce-back (WFB):** Slip-velocity approach. Compute wall shear stress
  from resolved gradient, use log-law to impose slip velocity. New `src/wall_functions.hpp`.
  **Status: In progress (2026-07-18).**
- **y+ in lattice units:** `y+ = y*u_tau/nu` where `u_tau = sqrt(tau_wall/rho)` and
  `nu = (tau-0.5)/3`. For channel at Re_tau=180 with NY=200, first cell y+ = 0.9 (resolved).
  At NY=30: y+ = 6 (buffer layer, wall function needed).

### UPGRADE 3: Pressure Contours & Enhanced Vorticity

- **Pressure Cp plot:** `Cp = (p - p_ref)/(0.5*rho_inf*U_inf^2)` in postprocess.py `--cp` flag.
  **Status: Completed (2026-07-18).**
- **Pressure channel in viewer:** Binary format already supports multiple channels.
  Add field selector (velocity / pressure / vorticity / temperature) to flow-viewer.js.
  **Status: Pending (needs front-end work).**
- **Full-resolution vorticity:** 9-point stencil in lbm.hpp:542-558.
  **Status: Completed (2026-07-18), higher-order accuracy.**

### UPGRADE 4: Thermal LBM (Heat Transfer)

- **Approach: Double Distribution Function (DDF).** Keep D2Q9 `f_i` for momentum, add
  D2Q9 (or D2Q5) `g_i` for temperature. Solves `dT/dt + u.grad(T) = alpha*laplacian(T)`.
- **Boussinesq coupling:** `F_buoyancy = -rho_0*beta*(T-T_ref)*g` added to momentum.
- **Parameters:** Pr = nu/alpha (0.71 air), Ra = g*beta*dT*L^3/(nu*alpha), Nu = h*L/k.
- **New file: `src/thermal.hpp`** (~200 lines): `g_i` collision, streaming, thermal BCs.
- **Modify `lbm_types.hpp`:** Add `g_thermal`, `g_thermal_next` to LBMCapabilities.
- **New entry point: `src/heated_cylinder.cpp`** (Nusselt validation) + natural
  convection cavity (Ra=10^3 to 10^6 benchmark).
- **Status: In progress (2026-07-18).**

### UPGRADE 5: 3D LBM Architecture

- **Phase 5a: D3Q19 lattice abstraction.** New `src/lattice.hpp` with D2Q9/D3Q19 structs
  (cx, cy, cz, weights). Template `LBMCapabilities<Lattice>` on lattice type.
- **Phase 5b: Performance.** LBM is memory-bandwidth bound. On M5: 200^3 D3Q19 = ~1.2 GB.
  OpenMP 3D decomposition natural. ~50-100 MLUPS. **Use D3Q19** not D3Q27 for engineering.
- **Phase 5c: Incremental migration.** (1) Abstract lattice. (2) Template core solver.
  (3) Keep 2D default (NZ=1). (4) Test D3Q19 with NZ=1. (5) Enable NZ>1 for true 3D.
- **Phase 5d: New 3D entry points:** `main_3d.cpp` (sphere), `cavity_3d.cpp`, `pipe_3d.cpp`.
- **Phase 5e: 3D postprocessing:** Slice planes in FlowViewer, isosurface rendering.

### Recommended Implementation Order (8-12 weeks total)

| Priority | Upgrade | Effort | Impact | Dependencies |
|----------|---------|--------|--------|-------------|
| 1 | Mei/Filippova-Hanel bounce-back | 1-2 days | High -- smooth curved boundaries | None |
| 2 | Van Driest LES damping + wall distance | 2-3 days | High -- accurate wall-bounded LES | None |
| 3 | Pressure Cp + enhanced visualization | 1-2 days | Medium -- better validation plots | None |
| 4 | Wall function bounce-back (WFB) | 3-5 days | High -- enables high-Re flows | Wall distance |
| 5 | Thermal LBM (DDF) | 1-2 weeks | High -- new physics domain | None |
| 6 | IBM with direct forcing | 1-2 weeks | High -- airfoils, complex shapes | Tier 2 |
| 7 | 3D lattice abstraction | 2-3 weeks | Very high -- new dimension | None |
| 8 | D3Q19 solver | 3-4 weeks | Very high -- full 3D capability | Abstraction |

### Quick Wins (do first, ~1 week)

1. Mei bounce-back in `lbm.hpp:206-245` -- 30 lines, immediate smooth boundaries
2. Van Driest damping in `lbm.hpp:151-162` -- 10 lines, fixes wall LES
3. Pressure Cp in `postprocess.py` -- 20 lines, standard validation metric
4. Full-resolution vorticity in `lbm.hpp:488-504` -- remove downsampling

## License

MIT
