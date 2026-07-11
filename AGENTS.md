# Project Context -- LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

## Style Rules
- **No em dashes** in any file. Use two regular hyphens (--) instead of ---, &mdash;, &ndash;, or literal Unicode em dash.
- **C++ code style**: 4-space indentation, K&R braces, no tabs, no trailing whitespace.
- **Variable naming**: snake_case for local variables, PascalCase for structs/classes, SCREAMING_SNAKE_CASE for constants.
- **HTML/CSS**: Double quotes for attributes, 2-space indentation, semantic HTML5 elements.

## Goal
Build and deploy a cache-optimized D2Q9 Lattice Boltzmann Method CFD solver in C++20 as a portfolio centrepiece for aerospace/defense engineering roles. Deliver an 8+ page HTML portfolio with per-case dedicated pages (interactive comparison sliders, KaTeX theory, validation tables), and a production-grade GitHub repository with CI and unit tests.

## Target Audience
Aerospace hiring managers at SpaceX, Firefly Aerospace, Lockheed Martin, Blue Origin, and similar. The site must communicate: HPC competency (C++, OpenMP, cache optimization), CFD fundamentals (MRT, Bouzidi, validation against benchmarks), and engineering communication skills (interactive web presentation, per-case analysis narratives).

## Current Status (2026-07-11)

| Phase | Description | Status |
|-------|-------------|--------|
| A | Grid resolution 2x all cases | ⏳ Not started |
| B | Polygon Bouzidi interpolated bounce-back | ⏳ Not started |
| C | Urban canyon: side-view + top-down + downwash (Oke 1988) | ⏳ Not started |
| D | Ribbed channel: friction factor + purpose docs | ⏳ Not started |
| E | Postprocessor updates | ⏳ Not started |
| F | Run all sweeps + generate images | ⏳ Not started |
| G | Website per-case dedicated pages + restructure | ⏳ Not started |

### Completed to date
- ✅ MRT collision operator (default, BGK fallback) with tuned rates
- ✅ Bouzidi interpolated bounce-back for cylinders (q_cylinder)
- ✅ JSON output pipeline (meta.json, forces.jsonl, frame_*.json)
- ✅ Backward-facing step, ribbed channel, urban canyon, downwash, Ahmed body cases
- ✅ Lid-driven cavity, NACA airfoil (legacy)
- ✅ Google Test suite (13 tests)
- ✅ GitHub Actions CI
- ✅ Postprocess.py with --split, --cmap, --strouhal, obstacle overlay, pressure contour
- ✅ Comparison slider (contour vs streamline), slider.js
- ✅ Old VTK files deleted, WASM archived, obsolete scripts removed
- ✅ Airfoil JSON output, urban top-down view
- ✅ Batch scripts for all cases

## Roadmap

### Phase A: Grid Resolution 2x (All Cases)

| Item | Current NX x NY | New NX x NY | Key dimension |
|------|----------------|-------------|---------------|
| Cylinder | 400 x 150 | 800 x 300 | D=60 (was 30) |
| Step | 400 x 150 | 800 x 300 | H_step=100 |
| Ribs | 200 x 100 | 400 x 200 | h_rib=10 |
| Urban (side) | 400 x 150 | 600 x 300 | H_bldg=60 |
| Urban (topdown) | 400 x 150 | 600 x 300 | W_bldg=30 |
| Downwash | 400 x 150 | 600 x 300 | H_tall=160 |
| Ahmed | 500 x 200 | 800 x 320 | H_body=64 |
| Airfoil | 400 x 150 | 800 x 300 | chord=160 |
| Cavity | 128 x 128 | 256 x 256 | standard |

Changes:
- `lbm_types.hpp`: Default NX=800, NY=300
- Each entry point can override via `NX = ...; NY = ...;`

### Phase B: Polygon Bouzidi Interpolated Bounce-Back

Current: `BounceBackGeometry` has `q_cylinder()` for circle boundaries. Need:
- `q_polygon()` using line-segment intersection for each polygon edge
- Called from `place_polygon()` and airfoil/Ahmed entry points
- Works for any closed polygon (airfoil, Ahmed body, arbitrary obstacles)

Implementation:
- Compute intersection of ray from (x,y) along direction i with each polygon edge
- Return smallest positive t <= 1.0 (same as q_cylinder)
- Store polygon vertices in BounceBackGeometry

### Phase C: Urban Canyon -- Comprehensive Oke 1988 Analysis

`urban_canyon.cpp` rewritten with `--mode` flag:

**`--mode side` (Side-View Canyon)**
- Two buildings forming a street canyon (cross-section view)
- Configurable H/W aspect ratio via `--ar 0.3|0.5|0.8`
- Flow from left, over buildings, recirculation in canyon
- Three Oke 1988 regimes: isolated roughness, wake interference, skimming flow
- Output: `output/urban_side_ar0.3/`, etc

**`--mode topdown` (Plan View)** -- existing implementation
- Two rectangular building footprints, channel walls at y=0, y=NY-1
- Street-level wind patterns around city blocks

**Sub-section on urban.html**: downwash results from `downwash.cpp`

### Phase D: Ribbed Channel Purpose + Friction Factor

Real-world applications to document on page:
- Gas turbine blade internal cooling (rib turbulators)
- Heat exchanger channels
- Electronics micro-channel cooling

Changes:
- Default contour: velocity magnitude (not pressure)
- Add friction factor computation: f = 2 * D_h * tau_wall / (rho * u_bulk^2)
- Compute wall shear stress from momentum exchange on ribs
- Compare to smooth-channel Blasius f = 64/Re
- Compute reattachment length Xr/h between ribs

### Phase E: Postprocessor Updates

- Handle urban side-view output directories
- Ribs: velocity contour + friction factor in meta.json
- Verify downsample scaling with new grid sizes
- Add `--friction` flag for ribbed channel analysis

### Phase F: Run All Sweeps + Generate Images

Full rerun at new resolutions:
- Cylinder: Re=20,40,100,200
- Step: Re=100,200,400
- Ribs: Re=50,100,200
- Urban side: ar=0.3,0.5,0.8 at Re=100
- Urban topdown: Re=100,200
- Downwash: Re=100,200
- Ahmed: Re=1000 at slant=25,30
- Airfoil: AoA=0,4,8,12,16 at Re=1000
- Cavity: Re=100,400,1000 at 256x256

### Phase G: Website Per-Case Dedicated Pages

```
docs/
  index.html              -- Project > Home (why build, compare to SU2/OF, case TOC)
  cylinder.html           -- Simulation > Cylinder
  cavity.html             -- Simulation > Lid-Driven Cavity
  step.html               -- Simulation > Backward Step
  airfoil.html            -- Simulation > Airfoil
  urban.html              -- Simulation > Urban Canyon (includes downwash section)
  ahmed.html              -- Simulation > Ahmed Body
  ribs.html               -- Simulation > Ribbed Channel
  theory.html             -- Reference (sidebar updated)
  implementation.html     -- Reference (sidebar updated)
  results.html            -- Redirect to index.html
```

Each case page has: field viewer (comparison slider for that case), validation tables, force plots, discussion.

## File Layout

```
lbm-2d/
  AGENTS.md                    # This file
  README.md                    # Project overview
  CMakeLists.txt               # Build system (OpenMP + Google Test)
  .gitignore

  src/
    lbm_types.hpp              # D2Q9 constants, MRT params, BounceBackGeometry,
                               # equilibrium, macros, index helpers
    lbm.hpp                    # Core solver: MRT collide + BGK, stream +
                               # Bouzidi BB, BCs, force extraction, JSON output
    geometry.hpp               # NACA 4-digit coords, polygon ops, point-in-polygon
    main.cpp                   # Cylinder flow entry point
    cavity.cpp                 # Lid-driven cavity entry point
    airfoil.cpp                # NACA 4-digit airfoil entry point (JSON output)
    step.cpp                   # Backward-facing step entry point
    ribs.cpp                   # Ribbed channel entry point
    urban_canyon.cpp           # Urban canyon (--mode side|topdown)
    downwash.cpp               # Building downwash entry point
    ahmed.cpp                  # Ahmed body entry point
    lbm_test.cpp               # Google Test suite (13 tests)
    archive/                   # Superseded code
      wasm_main.cpp, viewer.js, wasm_sim.js

  scripts/
    postprocess.py             # JSON -> PNG with --split, --cmap, --strouhal
    plot_airfoil.py            # Airfoil validation plots
    run_step.sh, run_ribs.sh, run_urban.sh, run_downwash.sh,
    run_ahmed.sh, run_all_cases.sh

  .github/workflows/
    ci.yml                     # GitHub Actions: build + test on ubuntu + macos

  docs/
    index.html                 # Home (to be rewritten)
    simulation.html            # Redirects to index.html
    theory.html                # LBM theory with KaTeX equations
    implementation.html        # Code architecture + source blocks
    results.html               # (to be deprecated -- redirect to index.html)
    css/style.css              # CFD Jet theme (dark, cyan/orange accents)
    assets/
      js/slider.js             # Comparison slider logic
      data/                    # Pre-computed JSON (per-case subdirectories)
      images/                  # Contour + streamline renders per case

  output/                      # JSON frames + forces (gitignored)
```

## D2Q9 Lattice Reference

Velocity vectors (cx[i], cy[i]):

```
Index: 0     1   2   3   4   5    6    7    8
cx:    0     1   0  -1   0   1   -1   -1    1
cy:    0     0   1   0  -1   1    1   -1   -1
```

Weights:
- w0 = 4/9   (rest)
- w1..w4 = 1/9   (axial)
- w5..w8 = 1/36  (diagonal)

3D index mapping for flat array:
- node_idx = y * NX + x
- dist_idx = (y * NX + x) * 9 + i

Inverse directions (bounce_back):
- bounce_back[i] gives index opposite to i: {0, 3, 4, 1, 2, 7, 8, 5, 6}

## Critical Knowledge

### Equilibrium distribution
f_i^eq = w_i * rho * (1 + 3*(c_i . u) + 4.5*(c_i . u)^2 - 1.5*(u . u))

### Viscosity relation
nu = c_s^2 * (tau - 0.5) * dt
With c_s^2 = 1/3, dt = 1, dx = 1:
nu = (tau - 0.5) / 3
Re = u_ref * L / nu                          (L = case-specific length scale)
tau = 0.5 + 3 * u_ref * L / Re

### MRT collision operator (default)
m = M * f       (9 moments: rho, e, epsilon, jx, qx, jy, qy, pxx, pxy)
m_i_new = m_i - s_i * (m_i - m_i_eq)   (s_i = relaxation rate per moment)
f_new = M^{-1} * m_new

Conserved moments (s_i = 0): rho, jx, jy
Shear moments: s_shear = 1/tau  (same as BGK viscosity)
Bulk moments: s_bulk = 1.2  (tuned for stability)
Normal moments: s_normal = 1.0  (relax fully)

### Interpolated bounce-back (Bouzidi 2001)
For curved boundaries, the wall is placed between lattice nodes:
  q = |x_f - x_w| / |x_f - x_b|   (normalized distance to boundary, 0 < q <= 1)
  q < 0.5:  f_{-i}(x_f) = 2q * f_i^*(x_f) + (1-2q) * f_i^*(x_f - e_i)
  q >= 0.5: f_{-i}(x_f) = (1/2q) * f_i^*(x_f) + (1 - 1/2q) * f_{-i}^*(x_f)

For polygons (airfoil, Ahmed body):
  q_polygon() computes intersection of ray (x,y)+t*e_i with polygon edges

### Momentum exchange method
F_i = e_i * [f(fluid, i) - f(obstacle, bounce_back(i))]
Cd = 2 * Fx / (rho * u^2 * L)
Cl = 2 * Fy / (rho * u^2 * L)

### Validation targets
Cylinder (Williamson 1988, Tritton 1959):
| Re | Strouhal | Cd_mean | Notes |
|----|----------|---------|-------|
| 20 | --       | ~2.0    | Steady, symmetric |
| 40 | --       | ~1.5    | Steady, recirculating wake |
| 60 | ~0.14    | ~1.4    | Onset of shedding |
| 100| 0.164-0.172 | ~1.4 | Fully periodic |
| 200| 0.180-0.195 | ~1.3 | Periodic, 3D effects begin |

Backward-facing step (Armaly 1983):
| Re_H | Xr/H (laminar) | Notes |
|------|-----------------|-------|
| 100  | ~3             | Steady, 2D |
| 200  | ~6             | Steady, 2D |
| 400  | ~9             | Near transition |

Urban canyon (Oke 1988):
| H/W  | Regime | Description |
|------|--------|-------------|
| <0.3 | Isolated roughness | Buildings independent |
| 0.3-0.65 | Wake interference | Wakes interact |
| >0.65 | Skimming flow | Single vortex in canyon |

### Grid sizing guide (new)
- Cylinder: NX = 800, NY = 300, D = NY/5 = 60
- Bwd Step: NX = 800, NY = 300, step at x = NX/4, H_step = NY/3
- Ribs: NX = 400, NY = 200, rib h = NY/20, pitch = 10*h
- Urban Canyon: NX = 600, NY = 300
- Downwash: NX = 600, NY = 300, tall H = 160
- Ahmed: NX = 800, NY = 320, body H = 64, L = 3.5*H
- Airfoil: NX = 800, NY = 300, chord = 160

## Color Scheme: CFD Jet Theme

```css
:root {
  --bg-primary:   #0d1117;
  --bg-card:      #161b22;
  --bg-canvas:    #0a0e14;
  --border:       #21262d;
  --cyan:         #00d4ff;
  --cyan-dim:     #0099cc;
  --orange:       #ff6b35;
  --turquoise:    #00f5d4;
  --green:        #39d353;
  --pink:         #ff79c6;
  --fg:           #c9d1d9;
  --fg-dim:       #8b949e;
  --fg-muted:     #484f58;
}
```

## Per-Case Colormap Guide

| Case | Primary colormap | Streamline color |
|------|-----------------|-------------------|
| Cylinder | jet (velocity) | jet |
| Lid-driven cavity | viridis (velocity) | viridis |
| Bwd step | coolwarm (vorticity) | viridis |
| Ribbed channel | plasma (velocity) | plasma |
| Urban canyon (side) | viridis (velocity) | magma |
| Urban canyon (topdown) | viridis (velocity) | magma |
| Building downwash | RdBu (pressure) | coolwarm |
| Ahmed body | jet (velocity) | jet |
| Airfoil | jet (velocity) | jet |

## JSON Output Format

```
output/{case}_re{Re}/
  meta.json:
    nx, ny, re, tau, u_inflow, shape_type, length_scale
    cd_mean, cl_amplitude, strouhal (filled by postprocess.py)

  forces.jsonl:
    {"step": 0, "cd": 0.0, "cl": 0.0}
    {"step": 100, "cd": 1.234, "cl": 0.056}
    ...

  frame_*.json:
    {"nx": 100, "ny": 38, "velocity": [...], "u": [...], "v": [...],
     "rho": [...], "obstacle": [...]}
    (downsampled 4x for web)
```

# Reference Commands

```bash
cmake -B build && cmake --build build          # Build solver (MRT default)

# Cylinder flow
./build/LBM_Engine                             # Re=100, 30000 steps (JSON output)
./build/LBM_Engine 200                         # Re=200

# Lid-driven cavity
./build/LBM_Cavity 100 256

# NACA airfoil
./build/LBM_Airfoil 0012 1000 0

# Backward-facing step
./build/LBM_Step 100

# Ribbed channel
./build/LBM_Ribs 100

# Urban canyon (side view)
./build/LBM_UrbanCanyon --mode side --ar 0.5

# Urban canyon (top-down view)
./build/LBM_UrbanCanyon 100

# Building downwash
./build/LBM_Downwash 100

# Ahmed body
./build/LBM_Ahmed 1000 30   # Re=1000, slant=30 deg

# Run tests
./build/LBM_Tests

# Post-processing
python3 scripts/postprocess.py output/step_re100 --split --cmap coolwarm
python3 scripts/postprocess.py output/step_re100 --strouhal

# Batch runners
bash scripts/run_step.sh
bash scripts/run_all_cases.sh

# Preview website
python3 -m http.server -d docs 8765
open http://localhost:8765
```

## CI Pipeline

`.github/workflows/ci.yml`:
- Trigger: push, pull_request to main
- Strategy matrix: ubuntu-latest, macos-latest
- Steps: checkout -> install OpenMP -> cmake configure -> cmake build -> run tests

## Known Issues

1. **MRT parameter tuning**: Relaxation rates s_e, s_eps, s_q, s_pi need tuning per Re for optimal stability. Defaults for Re=100 may not extend to Re=1000 without adjustment.
2. **Polygon Bouzidi**: q_polygon() detects intersection but cannot distinguish inside/outside the polygon; works for convex polygons where the ray exits through one edge. Concave polygon or complex shapes may produce incorrect q.
3. **2D Ahmed body**: The real Ahmed body flow is 3D with strong longitudinal vortices at the slant. A 2D slice captures only the base pressure drag trend, not the characteristic Cd drop at 25 deg.
4. **Urban canyon at low Re**: LBM-appropriate Re values (100-200) give laminar urban flow. Real urban flows are turbulent (Re > 10^6). Pedestrian-level patterns remain qualitatively correct.

## Solver Improvement Plan

Audit conducted 2026-07-11. Priorities organized by impact on solver accuracy, performance, portfolio quality.

### Priority A: Correctness & Accuracy (Fix Immediately)

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| 1 | MRT s_shear clamped to [0.5, 1.8] -- high-Re cases exceed upper bound, making effective viscosity too high | `lbm_types.hpp:63` | Widen clamp range or remove with stability check |
| 2 | Downwash missing top wall -- no obstacle at y=NY-1, top boundary defaults to periodic (unphysical) | `downwash.cpp` | Add `if (y == NY - 1) obst = true` |
| 3 | Ribs friction factor fields (`friction_factor`, `f_smooth`, `xr_h`) not persisted to meta.json | `ribs.cpp` | Debug meta.json append -- likely seekp position issue |
| 4 | Ribs Blasius comparison uses 64/Re (pipe) instead of 96/Re (channel) | `ribs.cpp` | Change constant |

### Priority B: Performance

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| 5 | `g_case` + `is_valid()` checked 9x per node per step in streaming inner loop | `lbm.hpp:264` | Hoist outside direction loop (~5 lines) |
| 6 | 4 full-grid traversals per step (collide, body force, zero f_next, stream+force) | `lbm.hpp` | Fuse force extraction into streaming pass |
| 7 | `std::vector<bool>` bit-packing overhead on obstacle hot path | `lbm_types.hpp:170` | Change to `std::vector<uint8_t>` |
| 8 | `compute_macros` called 4x per downsampled node in `save_json_frame` | `lbm.hpp:409` | Call once, reuse u/v/rho |
| 9 | `collapse(2)` `omp parallel` wastes threads on obstacle rows | `lbm.hpp` | Precompute fluid node index list at init |

### Priority C: Code Quality & Maintainability

| # | Issue | Effort | Fix |
|---|-------|--------|-----|
| 10 | ~400 lines of identical simulation loop across 8 entry points | 1-2 days | Extract `run_simulation()` template into lbm.hpp |
| 11 | `g_case = CaseType::CYLINDER` for airfoil (semantically wrong) | Small | Add `CaseType::AIRFOIL`, wire BCs |
| 12 | `::system("mkdir -p")` instead of `std::filesystem::create_directories` | Small | Replace in save_json_frame |
| 13 | Dead code: `n_cyl_nodes`, `dist_index()`, `write_json_double_array` | Small | Remove |
| 14 | Inconsistent naming: `fx_cyl`/`fy_cyl` used for all obstacle types | Small | Rename to `fx_body`/`fy_body` |

### Priority D: Features (Portfolio Polish)

| # | Feature | Effort | Details |
|---|---------|--------|---------|
| 15 | Convergence detection | Medium | Stop early when Cd changes < 1e-4 over 1000 steps |
| 16 | Pressure output (p = rho/3) in JSON frames | Small | 3 lines in save_json_frame |
| 17 | Vorticity output (omega = dv/dx - du/dy) | Small | Finite differences in save_json_frame |
| 18 | Command-line grid override (--nx --ny) | Medium | Shared arg parsing for all entry points |
| 19 | Checkpoint/restart | Low | Binary dump/load of f array |
| 20 | --vorticity flag in postprocess.py ([NYI]) | Small | ~50 lines in postprocess.py |

### Priority E: Stretch Goals (Beyond 2D)

| # | Feature | Effort | Reference |
|---|---------|--------|-----------|
| 21 | Smagorinsky LES turbulence model | 1-2 weeks | nu_t = (Cs*delta)^2 * |S|, enables turbulent urban Re > 10^6 |
| 22 | D3Q19 extension (3D lattice, 19 velocities) | 1-2 months | Required for Ahmed spanwise vortices, urban corner flows |
| 23 | AMR (Adaptive Mesh Refinement) | 2-4 weeks | Refine near obstacles/wakes, coarsen in free stream |

### Recommended Sequence

1. **Week 1**: A1-A4 + B5 (fix correctness, then rerun all simulations)
2. **Week 2**: B6-B7 + C10 + C12 (performance + code quality)
3. **Week 3**: D15-D18 + C13 (features + cleanup)
4. **Stretch**: E21 + D20 (LES model + vorticity postprocessor)
