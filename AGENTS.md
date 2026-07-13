# Project Context, LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

## Style Rules
- **No em dashes** in any file. Use two regular hyphens (--) instead of ---, &mdash;, &ndash;, or literal Unicode em dash.
- **C++ code style**: 4-space indentation, K&R braces, no tabs, no trailing whitespace.
- **Variable naming**: snake_case for local variables, PascalCase for structs/classes, SCREAMING_SNAKE_CASE for constants.
- **HTML/CSS**: Double quotes for attributes, 2-space indentation, semantic HTML5 elements.

## Goal
Build and deploy a cache-optimized D2Q9 Lattice Boltzmann Method CFD solver in C++20 as a portfolio centrepiece for aerospace/defense engineering roles. Deliver an 8+ page HTML portfolio with per-case dedicated pages (interactive comparison sliders, KaTeX theory, validation tables), and a production-grade GitHub repository with CI and unit tests.

## Target Audience
Aerospace hiring managers at SpaceX, Firefly Aerospace, Lockheed Martin, Blue Origin, and similar. The site must communicate: HPC competency (C++, OpenMP, cache optimization), CFD fundamentals (MRT, Bouzidi, Smagorinsky LES, AMR), and engineering communication skills (interactive web presentation, per-case analysis narratives).

## Current Status (2026-07-12)

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Solver Improvement Plan (correctness + perf + cleanup) | Completed |
| 1 | Smagorinsky LES turbulence model | Completed |
| 2 | Block-structured AMR (adaptive mesh refinement) | In progress |
| 3 | Vorticity output + postprocessor | Completed |
| 4 | Full simulation re-runs + new cases | In progress |
| 5 | Website updates for new features | Pending |

### Completed to date
- MRT collision operator (default, BGK fallback) with tuned rates
- Bouzidi interpolated bounce-back for cylinders (q_cylinder) and polygons (q_polygon)
- JSON output pipeline (meta.json, forces.jsonl, frame_*.json) with pressure field
- 15 simulation cases: cylinder, cavity, step, ribs, urban (side+topdown), downwash, square cylinder, flat plate, nozzle, periodic hills, cylinder near wall, side-by-side cylinders, rotating cylinder
- 12 Google Test suite tests, GitHub Actions CI
- Postprocess.py with --split, --cmap, --strouhal, --video, obstacle overlay, pressure contour
- Comparison slider (contour vs streamline), slider.js
- Solver Improvement Plan items A1-A4, B5-B8, C11-C14, D15-D16 all implemented
- forces.jsonl static cache fix (no more append corruption)
- Convergence detection function (check_convergence)
- `std::vector<bool>` -> `std::vector<uint8_t>`, g_case hoisted, body force fused into collision
- `fx_cyl`/`fy_cyl` -> `fx_body`/`fy_body`, removed dead code (dist_index, n_cyl_nodes, write_json_double_array)
- `system("mkdir -p")` -> `std::filesystem::create_directories` throughout
- MRT s_shear clamp widened to [0.5, 1.99]
- Downwash top wall fix, ribs meta.json seekp bug fix, Blasius 64->96 fix
- TECHNICAL_REPORT.md (947-line comprehensive technical document)
- Smagorinsky LES: g_use_les/g_cs globals, tau_eff quadratic solver, s_shear scaling in MRT collision
- Block-structured AMR: AMRBlock/AMRGrid structs, prolongation, restriction, refinement sensor, regridding
- Vorticity output: omega field in frame JSON, --vorticity flag in postprocess.py with RdBu colormap
- Square cylinder case (ERCOFTAC 043): sharp-edge separation, fixed separation points
- Urban canyon enhanced: 3 buildings in row, wider domain (900x400), larger buildings (H=NY/4, W=1.5H)
- Urban topdown enhanced: 3 buildings, larger (w=60, l=NY/2), wider spacing (2*w) for realistic street network
- Downwash enhanced: buildings scaled up (h_tall=120, h_low=45, w=45), maintaining 2.67 height ratio
- Cavity updated: JSON frame output added alongside VTK
- Auto-LES: automatically enables Smagorinsky when tau < 0.55 (high Re stability)
- Periodic hills: canonical LES benchmark (Moser/Kim/Moin 1993), sinusoidal hill profile
- Cylinder near wall: ground effect study with variable wall gap (2,4,8 cells)
- Side-by-side cylinders: interference study with variable S/D ratio (2,3,5)
- Rotating cylinder: Magnus effect with variable angular velocity (0.5,1.0,2.0 rad/ts)
- Removed: ahmed body (2D limitation), airfoil (replaced by flat plate), tandem cylinders (redundant)

### Simulation Results (Phase 4)

| Case | Re | Cd | Cl | Status |
|------|-----|-----|-----|--------|
| Flat plate AoA=0 | 1000 | ~0.026 | 0 | Pending (key validation) |
| Cylinder | 100 | 1.774 | ~0 | Validated |
| Cylinder | 200 | 1.495 | ~0 | Validated |
| Square cylinder | 200 | 1.157 | 0.47 | Validated vs ERCOFTAC |
| Cavity | 100-1000 |, |, | Validated vs Ghia |
| Step | 100-400 |, |, | Validated vs Armaly |
| Ribs | 50-200 | 0.26-0.64 | 0 | Validated |
| Nozzle (AR=0.25) | 100-1000 |, |, | Bernoulli validation |
| Periodic hills | 100-2800 |, |, | LES benchmark |
| Cylinder near wall | 100 |, |, | Ground effect study |
| Side-by-side | 100 |, |, | Interference study |
| Rotating cylinder | 100 |, |, | Magnus effect |
| Urban canyon | 100 |, |, | Oke 1988 regimes |
| Downwash | 100 |, |, | Hunt 1984 |

### Failed / Known Issues
- Cylinder Re=1000: NaN divergence at tau=0.518 without LES (auto-LES fix applied)
- Jeffery-Hamel: Removed (wedge apex singularity). Replaced by converging-diverging nozzle.

## Roadmap

### Phase 1: Smagorinsky LES Turbulence Model

Implements the Smagorinsky subgrid-scale eddy viscosity model for the MRT collision operator.

**Theory:**
- Eddy viscosity: nu_t = (Cs * delta)^2 * |S|
- Strain rate |S| from non-equilibrium stress moments:
  - pi_xx_neq = pxx - rho*(u*u - v*v)  (pre-relaxation normal stress)
  - pi_xy_neq = pxy - rho*u*v            (pre-relaxation shear stress)
  - Q = sqrt(2 * (pi_xx_neq^2 + pi_xy_neq^2))
- tau_eff from quadratic formula (resolves circular tau-|S| dependence):
  - A = 9 * Cs^2 * Q / (2 * rho)
  - tau_eff = (tau + sqrt(tau^2 + 4*A)) / 2
- Only s_shear changes: s_shear = 1/tau_eff; s_bulk, s_normal unchanged

**Changes:**
| File | Change |
|------|--------|
| `lbm_types.hpp` | Add `g_use_les`, `g_cs` globals |
| `lbm.hpp` | Compute Q in MRT collision loop, derive tau_eff, pass to MRTParams |

**Verification tests (short runs, ~500 steps each):**
| T | Re | LES | Expected |
|---|----|-----|----------|
| T1 | 100 | off | Baseline Cd/St |
| T2 | 100 | on | Identical to T1 (nu_t << nu) |
| T3 | 1000 | off | NaN divergence (current bug) |
| T4 | 1000 | on | Stable, Cd within 30% of turbulent literature |
| T5 | 200 | on/off | Slight difference, LES adds small eddy viscosity |

### Phase 2: Block-Structured AMR (Adaptive Mesh Refinement)

Full dynamic block-structured AMR with 2 levels (extensible to N levels). Refinement factor of 2 per level. Same dt for all levels (LBM CFL << 1 even on fine grids).

**Architecture:**
```
AMRGrid
  Level 0 (coarse base)
    Blocks[0..N0] each: nx, ny, f, f_next, obstacle, fx_body, fy_body, bb_geom
  Level 1 (2x finer)
    Blocks[0..N1] each: 2x resolution, dx=0.5, ghost layer for coarse-fine coupling
```

**Key algorithms:**
1. **Refinement sensor**: `sensor = |grad u| * dx = sqrt(du_dx^2 + du_dy^2 + dv_dx^2 + dv_dy^2) * dx`
2. **Block clustering**: Tile domain into 16x16 tiles, mark tiles containing sensor > threshold, merge adjacent marked tiles into rectangular blocks, expand by ghost cell layer
3. **Prolongation (coarse->fine)**: Bilinear interpolation of distribution functions from coarse parent nodes to fine ghost nodes
4. **Restriction (fine->coarse)**: Arithmetic average of 2x2 fine node f_i values onto coarse child node
5. **Regridding**: Every N_regrid steps (e.g., 100), rebuild block hierarchy from scratch

**New file:** `src/amr.hpp` (~755 lines)

**Verification tests (short runs):**
| A | Description | Steps | Check |
|---|-------------|-------|-------|
| A1 | Single level (AMR disabled) | 1000 | Matches non-AMR baseline |
| A2 | Static fine block around cylinder | 1000 | Cd within 1% of A1 |
| A3 | Dynamic refinement (vorticity sensor) | 2000 | Blocks follow wake, Cd stable |
| A4 | Regrid every 100 steps | 3000 | Block creation/destruction clean |
| A5 | 2 levels + LES at Re=1000 | 2000 | Stable, no NaN |
| A6 | Airfoil with AMR | 1000 | Smoother surface, Cd stable |

### Phase 3: Vorticity Output + Postprocessor

**C++ side:**
- Add `"omega"` array to frame JSON in save_json_frame
- Compute on downsampled grid using central differences: omega[i,j] = (v[i+1,j] - v[i-1,j])/(2*ds) - (u[i,j+1] - u[i,j-1])/(2*ds)

**Python side:**
- Wire up existing `--vorticity` flag (currently marked [NYI])
- When set: read `omega` field, render contour with RdBu symmetric colormap

### Phase 4: Full Simulation Re-Runs + New Cases

**New cases implemented:**
- Flat plate boundary layer: AoA sweep (-10 to 15 deg) + Re sweep (500-2000)
- Square cylinder (ERCOFTAC 043): Cd/St validation
- Converging-diverging nozzle: area ratio sweep, Bernoulli validation
- Periodic hills: canonical LES benchmark (Moser/Kim/Moin 1993)
- Cylinder near wall: ground effect study with variable wall gap
- Side-by-side cylinders: interference study with variable S/D ratio
- Rotating cylinder: Magnus effect with variable angular velocity

**Removed cases:**
- Ahmed body: 2D limitation, doesn't capture 3D slant vortices
- Airfoil: replaced by flat plate as primary validation case
- Tandem cylinders: redundant with urban topdown multi-body flow
- Jeffery-Hamel: wedge apex singularity, replaced by CD nozzle

**Re-runs with omega field:**
- All existing cases regenerated with vorticity output

**Auto-LES:**
- `src/main.cpp`: When tau < 0.55, automatically enable Smagorinsky LES
- Stabilizes high-Re cases (cylinder Re=1000) without user intervention

### Phase 5: Website Updates

- Add flat plate as PRIMARY validation case (first in navigation)
- Add square cylinder, nozzle, periodic hills, cylinder near wall, side-by-side, rotating cylinder pages
- Update sidebar navigation across all pages
- Add teaser cards on index.html

## File Layout

```
lbm-2d/
  AGENTS.md                    # This file
  README.md                    # Project overview
  TECHNICAL_REPORT.md          # Full technical report (947 lines)
  CMakeLists.txt               # Build system (OpenMP + Google Test)
  .gitignore

  src/
    lbm_types.hpp              # D2Q9 constants, MRT params, BounceBackGeometry,
                               # equilibrium, macros, index helpers, LES globals
    lbm.hpp                    # Core solver: MRT collide + LES, stream +
                               # Bouzidi BB, BCs, force extraction, JSON output
    geometry.hpp               # NACA 4-digit coords, polygon ops, point-in-polygon
    amr.hpp                    # AMRBlock, AMRGrid, refinement, regridding
    main.cpp                   # Cylinder flow entry point (+ auto-LES)
    flat_plate.cpp             # Flat plate boundary layer (PRIMARY validation)
    cavity.cpp                 # Lid-driven cavity entry point
    step.cpp                   # Backward-facing step entry point
    square_cylinder.cpp        # Square cylinder (ERCOFTAC 043), sharp-edge separation
    nozzle.cpp                 # Converging-diverging nozzle, Bernoulli validation
    ribs.cpp                   # Ribbed channel entry point
    urban_canyon.cpp           # Urban canyon (--mode side|topdown), 3 buildings
    downwash.cpp               # Building downwash entry point (scaled up buildings)
    periodic_hills.cpp         # Periodic hills (canonical LES benchmark)
    cylinder_near_wall.cpp     # Cylinder near wall (ground effect)
    side_by_side_cylinders.cpp # Side-by-side cylinders (interference)
    rotating_cylinder.cpp      # Rotating cylinder (Magnus effect)
    lbm_test.cpp               # Google Test suite (12 tests)
    archive/                   # Superseded code
      wasm_main.cpp, viewer.js, wasm_sim.js

  scripts/
    postprocess.py             # JSON -> PNG with --split, --cmap, --strouhal, --vorticity
    run_*.sh                   # Batch sweep scripts

  .github/workflows/
    ci.yml                     # GitHub Actions: build + test on ubuntu + macos

  docs/
    flat_plate.html            # PRIMARY validation case (Blasius, drag polar)
    cylinder.html              # Cylinder wake (comparison slider)
    square_cylinder.html       # ERCOFTAC 043 (sharp-edge separation)
    cavity.html                # Lid-driven cavity
    step.html                  # Backward-facing step
    ribs.html                  # Ribbed channel
    nozzle.html                # CD nozzle (Bernoulli validation)
    urban.html                 # Urban canyon (side + topdown + downwash)
    periodic_hills.html        # Periodic hills (LES benchmark)
    cylinder_near_wall.html    # Cylinder near wall (ground effect)
    side_by_side.html          # Side-by-side cylinders (interference)
    rotating_cylinder.html     # Rotating cylinder (Magnus effect)
    theory.html, implementation.html, index.html
    css/style.css              # CFD Jet theme (dark, cyan/orange accents)
    assets/
      js/slider.js             # Comparison slider logic
      data/                    # Pre-computed JSON (per-case subdirectories)
      images/                  # Contour + streamline renders per case

  output/                      # JSON frames + forces (gitignored)
```
    theory.html, implementation.html, index.html
    css/style.css              # CFD Jet theme (dark, cyan/orange accents)
    assets/
      js/slider.js             # Comparison slider logic
      data/                    # Pre-computed JSON (per-case subdirectories)
      images/                  # Contour + streamline renders per case

  output/                      # JSON frames + forces (gitignored)
```

## Smagorinsky LES Reference

### Eddy viscosity model
```
nu_t = (Cs * delta)^2 * |S|
tau_eff = tau + 3 * nu_t
s_shear_eff = 1 / tau_eff
```

### Strain rate from non-equilibrium moments (D2Q9 MRT)
```
pi_xx_neq = pxx - rho*(u*u - v*v)
pi_xy_neq = pxy - rho*u*v
Q = sqrt(2 * (pi_xx_neq^2 + pi_xy_neq^2))
|S| = 3 * Q / (2 * rho * tau_eff)
```

### Quadratic for tau_eff (circular dependence resolved analytically)
```
A = 9 * Cs^2 * Q / (2 * rho)
tau_eff = (tau + sqrt(tau^2 + 4*A)) / 2
```

### Typical values
- Smagorinsky constant Cs: 0.1-0.2 (default 0.12)
- Grid spacing delta: 1.0 (lattice units)
- Cs^2 = 0.0144 (at Cs=0.12)
- Auto-LES threshold: tau < 0.55 (Re > ~500)

## Block-Structured AMR Reference

### Grid hierarchy
```
Level 0: NX x NY, dx=1, dt=1, base level (always present)
Level 1: 2*NX x 2*NY per block, dx=0.5, dt=1 (refinement factor 2)
Level l: 2^l * NX x 2^l * NY per block, dx=1/2^l, dt=1
```

### Prolongation (coarse -> fine ghost nodes)
```
For each fine ghost node (xf, yf):
  1. Map to coarse coordinates: (xc, yc) = (xf/2, yf/2)
  2. Find 4 surrounding coarse nodes: (x0,y0), (x1,y0), (x0,y1), (x1,y1)
  3. Bilinearly interpolate f_i from coarse to fine position
  4. Assign to fine ghost node
```

### Restriction (fine -> coarse)
```
For each coarse node (xc, yc) covered by a fine block:
  1. Find 4 fine children: (2*xc, 2*yc), (2*xc+1, 2*yc), (2*xc, 2*yc+1), (2*xc+1, 2*yc+1)
  2. Average: f_coarse[i] = sum(f_fine[children][i]) / 4
  3. Replace coarse node's f_i with the average
```

### Refinement sensor (vorticity/gradient based)
```
sensor = sqrt(du_dx^2 + du_dy^2 + dv_dx^2 + dv_dy^2) * dx
  - threshold for refinement:  0.001 - 0.01 (case-dependent)
  - threshold for coarsening:   sensor/10
  - computed on the finest existing level
  - clustered into 16x16 tiles for block generation
```

### Regridding algorithm
```
1. Every N_regrid steps (default 100):
   a. Evaluate sensor on current finest level
   b. Tag cells: sensor > thresh_ref -> refine; sensor < thresh_coarse -> coarsen
   c. Enforce constraints: level 0 always present, min block size 16x16
   d. Cluster tagged cells into tile grid (16x16 tiles)
   e. Merge adjacent marked tiles into rectangular blocks
   f. Expand blocks by ghost_width + 1
   g. Build new block list for each level
   h. Interpolate f from old hierarchy to new blocks
   i. Delete old block hierarchy
```

## Known Issues

1. **MRT parameter tuning**: Relaxation rates s_e, s_eps, s_q, s_pi need tuning per Re for optimal stability. Defaults for Re=100 may not extend to Re=1000 without adjustment (mitigated by LES).
2. **Polygon Bouzidi**: q_polygon() detects intersection but cannot distinguish inside/outside the polygon; works for convex polygons where the ray exits through one edge. Concave polygon or complex shapes may produce incorrect q.
3. **2D Ahmed body**: The real Ahmed body flow is 3D with strong longitudinal vortices at the slant. A 2D slice captures only the base pressure drag trend, not the characteristic Cd drop at 25 deg.
4. **AMR coarse-fine interface**: Prolongation/restriction operators may produce spurious reflections at block boundaries. Mitigated by ghost cell overlap and validation against single-grid baseline.
5. **LES at very low Re**: Smagorinsky adds negligible eddy viscosity at Re < 200. Should match non-LES baseline within 0.1%.
6. **forces.jsonl**: Static cache approach assumes single output directory per process. Safe for all batch scripts (each is a separate process).

## Implementation Sequence

1. **Phase 1: Smagorinsky LES**, Completed
2. **Phase 2: Block-Structured AMR**, In progress (restriction operator needs fix)
3. **Phase 3: Vorticity + Postprocessor**, Completed
4. **Phase 4: Full Re-Runs + New Cases**, In progress (17 simulations pending)
5. **Phase 5: Website Updates**, Pending (4 new pages + navigation updates)
