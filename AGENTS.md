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
- 14 simulation cases: cylinder, cavity, step, urban (side+topdown), downwash, square cylinder, flat plate, orifice plate, periodic hills, cylinder near wall, side-by-side cylinders, rotating cylinder
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
- Cavity: JSON frame output; website uses JSON slider (static PNG gallery removed), VTK still written for Paraview
- Auto-LES: automatically enables Smagorinsky when tau < 0.55 (high Re stability)
- Periodic hills: canonical LES benchmark (Moser/Kim/Moin 1993), sinusoidal hill profile
- Cylinder near wall: ground effect study, cylinder raised to wall gaps 10/20/40 cells (clear under-flow)
- Side-by-side cylinders: TRANSVERSE arrangement (same x, offset in y), D=40 (NY/15) so S/D=5 fits domain, S/D=2,3,5
- Rotating cylinder: Magnus effect with variable angular velocity (0.5,1.0,2.0 rad/ts)
- Ladd (1994) moving boundary: omega_lat = omega_user * u_inflow / R, f_bb correction term
- Cylinder near wall: physical wall at y=0, force extraction filtered to cylinder only
- Orifice plate: 4 configs (1p1h, 1p3h, 2p, 3p); u_inflow=0.025 + LES for single-hole jet stability
- Periodic hills: hill height reduced to h_max=H/6 (was H/2); L changed from 9*H=1800 to NX=800 so exactly one hill period fits the domain (fixes the periodic-x BC height discontinuity that caused NaN divergence at step 2); fully-periodic streaming + force-extraction paths wired for PERIODIC_HILLS
- Urban canyon: moved to External Aerodynamics nav. Side view = 4 cases (H/W 0.3/0.5/0.8 with 2 buildings, H/W 0.6 with 3 buildings). Top-down adds horizontal-orientation buildings (wind funneled along pedestrian, orifice-like) alongside vertical
- Removed: nozzle (replaced by orifice plate), ahmed body (2D limitation), airfoil (replaced by flat plate), tandem cylinders (redundant), sports-bell (removed from portfolio to focus on validated cases)
- Website: removed stats-bar from all case pages (data moved to setup/validation tables), "Field Viewer" renamed "Velocity Field" with explanatory text, deleted orphaned results.html/simulation.html
- PINN surrogate suite (Phase 6.0-6.2): `pinn/` directory with torch-free config/loader, PINN MLP (64x8, tanh), hybrid loss (PDE residual + data + BC), MPS training on Apple Silicon, 3-panel comparison (LBM/PINN/Error), website integration on cylinder.html
- PINN training: Cylinder Re=100 steady-state, 15k Adam epochs + cosine annealing, importance sampling near cylinder, L2 u=36.3% vs LBM baseline
- PINN parametric cavity (Phase 6.3): `ParametricPINN` (spatial + Re_n input), multi-Re training (Re=100+400), importance-sampled sensors (3000), hybrid loss with pressure (data_loss_full), v2 trained (462K params, HIDDEN=256, N_LAYERS=8) -- u L2 56.1%/41.8%, v L2 34.7%/33.0%, p L2 12.6%/12.6% for Re=100/400
- PINN Fourier feature layer (`FourierFeatureLayer`, in progress): frozen random sinusoidal projection (m=128, sigma=5.0) applied to spatial coords only, lifting (x,y) to 512-dim frequency space to break tanh spectral bias; MLP input becomes 513-dim (512 fourier + 1 Re_n)

### Simulation Results (Phase 4)

| Case | Re | Cd | Cl | Status |
|------|-----|-----|-----|--------|
| Flat plate AoA=0 | 1000 | ~0.026 | 0 | Pending (key validation) |
| Cylinder | 100 | 1.774 | ~0 | Validated |
| Cylinder | 200 | 1.495 | ~0 | Validated |
| Square cylinder | 200 | 1.157 | 0.47 | Validated vs ERCOFTAC |
| Cavity | 100-1000 |, |, | Validated vs Ghia |
| Step | 100-400 |, |, | Validated vs Armaly |
| Orifice plate | 100 | Fx 0.9 (1p3h) to 63 (3p) |, | ISO 5167 loss validation |
| Periodic hills | 100-2800 |, |, | LES benchmark (re-run pending after L=NX fix) |
| Cylinder near wall | 100 | 2.56-2.75 | +0.40 to +1.42 | Ground effect (lift vs gap) |
| Side-by-side | 100 | 2.57-2.82 | ~0 (amp 0.6-0.7) | Interference study |
| Rotating cylinder | 100 | Cd~2-7, Cl~-1.5 to -7.4 |, | Magnus effect (Ladd) |
| Urban canyon | 100 | Cd 0.37 (AR0.3) to 55 (topdown) | Cl 6.9 (AR0.5) to 20.2 (AR0.8) | Oke 1988 regimes; topdown vertical vs horizontal |
| Downwash | 100 |, |, | Hunt 1984 |

### Failed / Known Issues
- Cylinder Re=1000: NaN divergence at tau=0.518 (auto-LES applied but tau too low for this grid, needs coarser nu or finer grid)
- Jeffery-Hamel: Removed (wedge apex singularity). Replaced by converging-diverging nozzle (now orifice plate).
- Rotating cylinder: Ladd (1994) implemented, Cl~50-60% of Kutta-Joukowski prediction (viscous effects)
- Cylinder near wall: Wall effect working, Cd~2.6 (vs 1.77 isolated), Cl +0.4 to +1.4 (upward, scales with gap)
- Orifice 3p: Diverged at step 58500 at 60k steps with u_inflow=0.025+LES; rerun at 50k steps completed clean.
- Periodic hills: L>NX periodic-x discontinuity fixed in code (L=NX, 3 hill cycles via n_cycles=3, fully-periodic streaming + force extraction). Body-force driving implemented (compute_body_force, ×28 safety factor); re-run Re=100 (40k steps), Re=1000/2800 (240k steps) completed clean, images regenerated, docs updated (3-cycle callout + Fx table).
- Flat plate AoA force extraction: at AoA=5 deg the computed streamwise Cd (0.046) is lower than at AoA=0 (0.105), which is physically wrong (drag should rise with incidence). The momentum-exchange extraction reports the global x/y force correctly, so this points to a drag/lift decomposition issue under rotation that needs review. AoA=0 (Re-sweep) and the Cd reference-area convention (Cd = 2·Cf on planform area) are validated.
- Urban topdown horizontal: 3 long slabs (long in x, stacked in y) develop shear-layer (Kelvin-Helmholtz) instability in the canyons that grows until NaN divergence (~step 40k) without dissipation. Forcing Smagorinsky LES (--use-les) stabilizes it; appropriate since the separated canyon flow is turbulent.

### Pending Fixes (Phase 4)

| Fix | Problem | Solution | Files | Priority |
|-----|---------|----------|-------|----------|
| Ladd moving boundary | Rotating cylinder has no tangential velocity (Cl~0) | Implement Ladd (1994) bounce-back with wall velocity: f_bb = f_opp - 2*w_i*rho*(e_i.u_wall)/c_s^2 | `lbm_types.hpp`, `lbm.hpp`, `rotating_cylinder.cpp` | **Completed** |
| Cylinder near wall | Wall not affecting flow (Cd identical to isolated cylinder) | Add obstacle nodes at y=0 to create physical wall, filter force extraction to cylinder only | `cylinder_near_wall.cpp`, `lbm.hpp` | **Completed** |
| Periodic hills re-run | Code fix applied (L=NX, periodic streaming); driving mechanism missing; images stale | Body-force driving implemented; 3 hill cycles (n_cycles=3); re-run Re=100/1000/2800, regenerated images, updated docs (3-cycle + Fx table) | `periodic_hills.cpp`, `lbm.hpp`, `docs/periodic_hills.html` | **Completed** |
| Cylinder Re=1000 | Diverged at step 16k on coarse 800x300 grid (tau=0.518 < 0.55) | Stable on fine grid (NX=2400, NY=900, tau=0.554, no LES) but unsteady (vortex shedding); 20000 steps not fully converged. Documented as known limitation; website surfaces Re=20/40/100/200 only | `main.cpp` (--nx/--ny override added) | **Deferred** |

## Roadmap

### Phase 6: Physics-Informed Neural Network (PINN) Surrogate Suite

**Goal:** Build a parametric PINN surrogate suite that generalizes across multiple LBM cases and physical parameters. Start with single-case steady-state (cylinder), then parametric (cavity Re-sweep), then multi-case (step, orifice geometry). This mirrors the SciML R&D pipeline at NASA, Rolls-Royce, and F1 teams: a high-performance physics engine generates baseline data, a PINN provides a real-time, deployable surrogate.

**Hardware:** Apple M5 MacBook Pro. Use `torch.device("mps")` for training. No CUDA.

**Parametric PINN architecture:** Pass geometric or fluid parameters directly into the network alongside space and time:
```
[x, y, Re, hole_w, n_plates, ...] --> [Neural Network] --> [u, v, p]
```
This transforms the network from a single-case calculator into a continuous design-space surrogate. A recruiter can drag a parameter slider and see the flow field update instantly.

**Code location:** `pinn/` directory. Zero changes to existing C++ solver.

**Phases / timeline:**

| Phase | Description | Status |
|-------|-------------|--------|
| 6.0 | Environment setup: `pinn/` dir, requirements.txt, data/loader.py | Completed |
| 6.1 | Steady-state hybrid PINN (Cylinder Re=100), 3-panel comparison PNG | Completed |
| 6.2 | Website: add PINN section + error delta map to cylinder.html | Completed |
| 6.3 | Cavity parametric PINN (x,y,Re) -> (u,v,p), Re=100/400 | **In progress** |
| 6.4 | Backward-facing step PINN (x,y,Re) -> (u,v,p) | Pending |
| 6.5 | Orifice plate parametric PINN (x,y,hole_w,n_plates) -> (u,v,p) | Pending |
| 6.6 | ONNX export + WASM real-time inference page | Pending |
| 6.7 | Ablation study (data/PDE/BC terms), methodology write-up | Pending |

#### Phase 6.3: Cavity Parametric PINN (FIRST)

**Why cavity first:** Zero new simulations needed. Data exists for Re=100 (steady, 51 frames at 128x128 with p+omega) and Re=400 (steady). Geometry is trivially simple (all no-slip walls + moving lid). Ghia 1982 validation data available. Lowest risk path to demonstrating parametric capability.

**Architecture:** Input extends from 2 to 3 dims: `(x, y, Re_n)` where `Re_n = (Re - 100) / 900` normalizes [100, 1000] to [0, 1]. Network is a `ParametricPINN`: spatial coords (x, y) pass through a frozen random Fourier feature layer (m=128, sigma=5.0) before the MLP, lifting inputs to a 512-dim frequency space to break tanh spectral bias. MLP has 256 hidden units, 8 layers (462K params). Physical params (Re_n) are concatenated after Fourier features (513-dim MLP input).

**Training strategy:** Multi-Re training: sample collocation points at both Re=100 and Re=400 in each epoch. Sensor data from both frames (importance-sampled, 3000 sensors). BC losses for walls + moving lid (BCs are Re-independent). Hybrid loss = w_pde*PDE + w_data*data(u,v,p) + w_bc*BC.

**Training results:**
- v1 (64-wide, 116K params, no Fourier): u L2 73.5%/52.6%, v L2 45.6%/38.8%, p L2 107%/108% (Re=100/400)
- v2 (256-wide, 462K params, no Fourier): u L2 56.1%/41.8%, v L2 34.7%/33.0%, p L2 12.6%/12.6%
- **Fourier feature (v3, in progress):** expected u L2 15-25%, p captures vortex pressure variation

**Key diagnostics:** v2 still shows spectral bias -- u_max predicted only 39% of true (0.038 vs 0.097), pressure span only 1.7% of true, error concentrated at lid boundary layer. L-BFGS fine-tune gave only 0.2% improvement, confirming the issue is model representation, not optimization.

**Parametric demo:** "Drag Re slider from 100 to 400, watch vortex center migrate from y/H ~ 0.70 to ~ 0.68."

#### Phase 6.4: Backward-Facing Step (SECOND)

**Data requirement:** Re-run Re=100 with p/omega output (current frames lack pressure). Code exists, ~2hr run. Use Re=400 data as fallback (has p/omega but under-converged).

**Architecture:** Same 3-input (x, y, Re_n) as cavity. Add parabolic inlet BC loss. Encode step geometry via obstacle mask.

**Parametric demo:** "Drag Re slider, watch reattachment length Xr/H grow from 2.4 to 4.0+."

#### Phase 6.5: Orifice Plate Parametric (THIRD)

**Data requirement:** New simulations needed -- Re sweep (50, 100, 200, 500) for 1p1h config, plus hole-width sweep (10, 20, 50, 80) at Re=100. Only 1p1h needed for initial training.

**Architecture:** Input: `(x, y, Re_n, hole_w_n)` where hole_w is normalized. Encode rectangular plate geometry analytically (step function in obstacle mask).

**Parametric demo:** "Drag orifice diameter slider, see loss coefficient K change in real-time." Highest portfolio impact.

**Steady-state PINN architecture:**
```
Input: (x, y) normalized to [-1, 1]
  FC 2 -> 64 (x8 hidden, tanh) -> 3
Output: (u, v, p)
```

**Hybrid loss:**
```
L_total = w_data * MSE(u_pred, u_lbm) + MSE(v_pred, v_lbm)
        + w_pde  * PDE_residual(u, v, p)
        + w_bc   * BC_loss(inflow=u_inflow, walls=no-slip, outlet=zero-grad)
```
PDE residual = steady incompressible Navier-Stokes (continuity + 2 momentum eqns), derivatives via torch.autograd.

**Files to create (no C++ changes):**
```
pinn/
  README.md              # Setup, architecture, usage
  requirements.txt       # torch, numpy, matplotlib, scipy, onnx, onnxruntime
  config.py              # Case params (nx, ny, obstacle, Re, tau, u_inflow)
  data/loader.py         # Read frame*.json + forces.jsonl -> numpy arrays
  models/pinn.py         # PINN MLP + forward
  models/losses.py       # PDE residual, BC loss, data loss
  train.py               # MPS training loop (Adam + L-BFGS)
  evaluate.py            # Inference on full grid -> numpy fields
  export_onnx.py         # torch.onnx.export -> model.onnx
  plot_results.py        # 3-panel (LBM / PINN / Error delta)
  wasm/static/infer.html # ONNX Runtime Web canvas inference
```

**Website integration:** Keep existing LBM slider (contour | streamline). Add below it a 3-panel static comparison (C++ LBM / PINN surrogate / absolute error delta). Later add a live ONNX Runtime Web inference page (PINN-only; no C++ WASM rebuild) with a frame slider and field selector.

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
- Orifice plate: 4 configs (1p1h, 1p3h, 2p, 3p), ISO 5167 loss validation
- Periodic hills: canonical LES benchmark (Moser/Kim/Moin 1993)
- Cylinder near wall: ground effect study with variable wall gap (10/20/40 cells)
- Side-by-side cylinders: transverse interference study with variable S/D ratio (2,3,5)
- Rotating cylinder: Magnus effect with variable angular velocity

**Removed cases:**
- Ahmed body: 2D limitation, doesn't capture 3D slant vortices
- Airfoil: replaced by flat plate as primary validation case
- Tandem cylinders: redundant with urban topdown multi-body flow
- Jeffery-Hamel: wedge apex singularity, replaced by CD nozzle (now orifice plate)
- Converging-diverging nozzle: replaced by orifice plate (single + multi-stage)

**Re-runs with omega field:**
- All existing cases regenerated with vorticity output

**Auto-LES:**
- `src/main.cpp`: When tau < 0.55, automatically enable Smagorinsky LES
- Stabilizes high-Re cases (cylinder Re=1000) without user intervention

### Phase 5: Website Updates

- Add flat plate as PRIMARY validation case (first in navigation)
- Add square cylinder, orifice plate, periodic hills, cylinder near wall, side-by-side, rotating cylinder pages
- Update sidebar navigation across all pages
- Add teaser cards on index.html
- Remove stats-bar across all case pages; rename "Field Viewer" -> "Velocity Field" with parameter + slider explanations
- Remove nozzle section entirely (source, html, cmake, enum, nav, images, postprocess); delete orphaned results.html/simulation.html

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
    square_cylinder.cpp          # Square cylinder (ERCOFTAC 043), sharp-edge separation
    orifice_plate.cpp            # Orifice plate (single + multi-stage, staggered)
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
    orifice_plate.html           # Orifice plate (single + multi-stage)
    urban.html                   # Urban canyon (side + topdown + downwash)
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

  pinn/                        # Physics-Informed Neural Network surrogate suite (NEW)
    README.md                  # Setup, architecture, roadmap
    requirements.txt           # torch, numpy, matplotlib, scipy, onnx, onnxruntime
    config.py                  # CaseConfig + convenience constructors (cavity, step, orifice)
    data/loader.py             # Read frame*.json + forces.jsonl -> numpy arrays
    models/pinn.py             # PINN + ParametricPINN MLP architectures
    models/losses.py           # PDE residual, BC loss (cavity/cylinder), data loss
    train.py                   # Cylinder Re=100 training loop
    train_cavity.py            # Cavity parametric PINN (single-Re + multi-Re)
    evaluate.py                # Inference on full grid -> numpy fields
    export_onnx.py             # torch.onnx.export -> model.onnx
    plot_results.py            # 3-panel (LBM / PINN / Error delta)
    wasm/static/infer.html     # ONNX Runtime Web canvas inference

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
