# Project Context, LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

## Style Rules
- **No em dashes** in any file. Use two regular hyphens (--) instead of ---, &mdash;, &ndash;, or literal Unicode em dash.
- **C++ code style**: 4-space indentation, K&R braces, no tabs, no trailing whitespace.
- **Variable naming**: snake_case for local variables, PascalCase for structs/classes, SCREAMING_SNAKE_CASE for constants.
- **HTML/CSS**: Double quotes for attributes, 2-space indentation, semantic HTML5 elements.
- **Colormap convention**: All velocity/flow field plots and animations use the `jet` (rainbow) colormap for visual consistency across the portfolio (decided over turbo on 2026-07-16). The 3-panel error-delta panel uses `Reds`. Vorticity uses `RdBu` (signed). `viridis`/`turbo` LUTs remain available in `colormaps.js` but are not the default.

## Goal
Build and deploy a cache-optimized D2Q9 Lattice Boltzmann Method CFD solver in C++20 as a portfolio centrepiece for aerospace/defense engineering roles. Deliver an 8+ page HTML portfolio with per-case dedicated pages (interactive comparison sliders, KaTeX theory, validation tables), and a production-grade GitHub repository with CI and unit tests.

## Target Audience
Aerospace hiring managers at SpaceX, Firefly Aerospace, Lockheed Martin, Blue Origin, and similar. The site must communicate: HPC competency (C++, OpenMP, cache optimization), CFD fundamentals (MRT, Bouzidi, Smagorinsky LES, AMR), and engineering communication skills (interactive web presentation, per-case analysis narratives).

## Current Status (2026-07-16)

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Solver Improvement Plan (correctness + perf + cleanup) | Completed |
| 1 | Smagorinsky LES turbulence model | Completed |
| 2 | Block-structured AMR (adaptive mesh refinement) | In progress (restriction operator needs fix) |
| 3 | Vorticity output + postprocessor | Completed |
| 4 | Full simulation re-runs + new cases | In progress (17 simulations pending) |
| 5 | Website updates for new features | In progress (interactive LBM/PINN viewers on all cases) |
| 5.5 | Cavity page deep dive + PINN surrogate narrative | Completed (Key Findings, LBM Analysis, Training Convergence, What PINN Unlocks, Limitations sections; loss + temporal L2 plots; 600x speed section; Re=300 interpolation; sensitivity map) |
| 6 | PINN surrogate suite (cavity steady + temporal) | Completed (Re=100/400/1000 temporal trained; Re=300 interpolation; ONNX + binary export) |
| 6.9 | Model improvement roadmap (pressure-Poisson, Re range) | Pending |

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
- Removed ribs case (deprecated) from nav and results; ribs.html deleted if present
- Removed: nozzle (replaced by orifice plate), ahmed body (2D limitation), airfoil (replaced by flat plate), tandem cylinders (redundant), sports-bell (removed from portfolio to focus on validated cases)
- Website: removed stats-bar from all case pages (data moved to setup/validation tables), "Field Viewer" renamed "Velocity Field" with explanatory text, deleted orphaned results.html/simulation.html
- PINN surrogate suite (Phase 6.0-6.2): `pinn/` directory with torch-free config/loader, PINN MLP (64x8, tanh), hybrid loss (PDE residual + data + BC), MPS training on Apple Silicon, 3-panel comparison (LBM/PINN/Error), website integration on cylinder.html
- PINN training: Cylinder Re=100 steady-state, 15k Adam epochs + cosine annealing, importance sampling near cylinder, L2 u=36.3% vs LBM baseline
- PINN parametric cavity (Phase 6.3): `ParametricPINN` (spatial + Re_n input), multi-Re training (Re=100+400), importance-sampled sensors (3000), hybrid loss with pressure (data_loss_full), v2 trained (462K params, HIDDEN=256, N_LAYERS=8) -- u L2 56.1%/41.8%, v L2 34.7%/33.0%, p L2 12.6%/12.6% for Re=100/400
- PINN Fourier feature layer (`FourierFeatureLayer`, done): frozen random sinusoidal projection (m=128, sigma=5.0) applied to spatial coords only, lifting (x,y) to 512-dim frequency space to break tanh spectral bias; MLP input becomes 513-dim (512 fourier + 1 Re_n); 593K params
- PINN cavity v3 (Fourier, done): u L2 23.7%/24.4%, v L2 29.3%/30.0%, p L2 12.5%/12.6% for Re=100/400; u_max ratio true 1.24 (was 3.50 in v2 -- spectral bias fixed); velocity L2-err 30x lower than v2; parametric Re=200 interpolation gives smooth vortex-center migration (y/H 0.64->0.58 from Re=100->400)
- PINN temporal (Phase 6.8, done): time-parametric `ParametricPINN(n_params=2)` trained on 51-frame sequences at Re=100/400, 593,155 params, 201 min MPS; u L2 ~33% mean over transient (final frame 29.9%/34.7%), v L2 ~43-48%; u_max ratio 1.13-1.16; ONNX + float16 frame binaries exported; cavity.html PINN Prediction section animates full transient via second FlowViewer
- PINN web engine (Phase 6.6, done): `flow-viewer.js` velocity-only canvas engine (jet colormap, flipY-aware streamlines), `pinn-inference.js` placeholder surrogate, vendored onnxruntime-web (numThreads=1); binary `.bin` (+.gz) viewer data exported for all 11 cases via `pinn/export/export_web_data.py`

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
| 6.3 | Cavity parametric PINN (x,y,Re) -> (u,v,p), Re=100/400 | **Completed** (velocity; pressure paused) |
| 6.4 | Backward-facing step PINN (x,y,Re) -> (u,v,p) | Pending |
| 6.5 | Orifice plate parametric PINN (x,y,hole_w,n_plates) -> (u,v,p) | Pending |
| 6.6 | ONNX export + WASM real-time inference page | **Completed** (temporal ONNX exported, vendored runtime) |
| 6.7 | Ablation study (data/PDE/BC terms), methodology write-up | Pending |
| 6.8 | Time-parametric PINN (spatio-temporal surrogate) | **Completed** (Re=100/400/1000 trained, ONNX + binary export done) |
| 6.8b | Extend temporal PINN to Re=1000 | **Completed** (trained, exported; Re=300 interpolation panel added) |
| 6.9 | Model improvement roadmap (pressure-Poisson, Re range, curriculum, etc.) | Pending |

#### Phase 6.3: Cavity Parametric PINN (FIRST)

**Why cavity first:** Zero new simulations needed. Data exists for Re=100 (steady, 51 frames at 128x128 with p+omega) and Re=400 (steady). Geometry is trivially simple (all no-slip walls + moving lid). Ghia 1982 validation data available. Lowest risk path to demonstrating parametric capability.

**Architecture:** Input extends from 2 to 3 dims: `(x, y, Re_n)` where `Re_n = (Re - 100) / 900` normalizes [100, 1000] to [0, 1]. Network is a `ParametricPINN`: spatial coords (x, y) pass through a frozen random Fourier feature layer (m=128, sigma=5.0) before the MLP, lifting inputs to a 512-dim frequency space to break tanh spectral bias. MLP has 256 hidden units, 8 layers (462K params). Physical params (Re_n) are concatenated after Fourier features (513-dim MLP input).

**Training strategy:** Multi-Re training: sample collocation points at both Re=100 and Re=400 in each epoch. Sensor data from both frames (importance-sampled, 3000 sensors). BC losses for walls + moving lid (BCs are Re-independent). Hybrid loss = w_pde*PDE + w_data*data(u,v,p) + w_bc*BC.

**Training results:**
- v1 (64-wide, 116K params, no Fourier): u L2 73.5%/52.6%, v L2 45.6%/38.8%, p L2 107%/108% (Re=100/400)
- v2 (256-wide, 462K params, no Fourier): u L2 56.1%/41.8%, v L2 34.7%/33.0%, p L2 12.6%/12.6%
- **v3 (Fourier, done):** 593K params (512 fourier + 1 Re_n input), 177min training -- u L2 23.7%/24.4%, v L2 29.3%/30.0%, p L2 12.5%/12.6% for Re=100/400

**Key diagnostics (v3 vs v2):**
- u_max ratio (pred/true) improved from 3.50 (v2, overshoot) to 1.24 (v3) at Re=100 -- spectral bias fixed
- velocity L2-err (field mean abs) dropped 30x: lid 0.182->0.006, core 0.154->0.006, bottom 0.174->0.003
- pressure still essentially CONSTANT: pred std=0.0015 vs true std=0.0413; the p L2=12.5% is misleading (just captures the mean, since pressure is mean-dominated). Root cause: p treated as independent output, not coupled to velocity via the pressure Poisson equation. Fix: add pressure-Poisson residual term to the loss (Phase 6.3d).
- parametric interpolation verified: Re=200 (between training points) gives smooth vortex-center migration y/H 0.64->0.58 from Re=100->400, u_max decreasing physically with Re.

**Parametric demo:** "Select Re=100, 400, or 1000 (discrete buttons) and watch vortex center migrate; Re=300 shown as an interpolated prediction panel."

**Website integration (done):** cavity.html updated with:
- Architecture table (Fourier features, MLP, params, training time)
- 3-panel comparison PNGs for Re=100, Re=400, Re=1000 (LBM / PINN in jet / Error delta in Reds)
- Parametric Re-sweep (Re=100/300/400/1000) with discrete Re buttons + interpolated Re=300 panel
- Accuracy summary table (L2 errors, u_max ratio, vortex center)
- Speed section: ~60-100 ms/surrogate frame vs ~30 s/LBM frame (~300-600x)
- Sensitivity map (∂u/∂Re via autograd) at Re=300
- Images: `docs/assets/images/cavity/pinn_comparison_re{100,400,1000}.png`, `pinn_parametric_re{100,300,400,1000}.png`

**Pressure note:** Pressure prediction is near-constant (std 0.0015 vs true 0.0413). Paused to focus on velocity (primary deliverable). See Phase 6.3d for future fix.

#### Phase 6.3d: Pressure-Poisson Coupling (FIX pressure, paused)

**Problem:** v3 predicts near-constant pressure (pred std=0.0015 vs true 0.0413). The p L2=12.5% is misleading -- it just captures the pressure mean (pressure is mean-dominated, so a constant prediction yields ~12.6% L2). Root cause: p is a decoupled network output; the momentum equations are satisfied by a near-constant p (pressure gradients are small in the bulk), so the PDE loss gives no incentive to learn pressure variation.

**Fix:** Add a pressure-Poisson residual to the hybrid loss. Taking the divergence of the steady momentum equation gives:
```
Laplacian(p) = -rho * ( d2(uu)/dx2 + 2*d2(uv)/dxdy + d2(vv)/dy2 )
```
Enforcing this couples p to the velocity field structure and forces the correct pressure variation. This also strengthens velocity-pressure consistency.

**Status:** Not yet implemented. Velocity (primary deliverable) is already excellent; pressure is secondary for the cavity Re-button demo. Candidate follow-up after Phase 6.4/6.5, or sooner if pressure field is needed for the 3-panel error plot.

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

**Files (reorganized, no C++ changes):**
```
pinn/
  README.md              # Setup, architecture, usage
  requirements.txt       # torch, numpy, matplotlib, scipy, onnx, onnxruntime
  cases/cavity/          # train_steady.py, train_temporal.py, export_sweep.py,
                         #   export_temporal.py, plot_results.py, logs/
  cases/cylinder/        # train.py, evaluate.py
  export/export_web_data.py  # LBM frames -> float16 .bin (gz), PINN sweep, ONNX model
  data/loader.py         # Read frame*.json + forces.jsonl -> numpy arrays
  data/temporal_loader.py# Load LBM frames -> (x,y,Re_n,t_n) -> (u,v,p) samples
  models/pinn.py         # PINN MLP + ParametricPINN + FourierFeatureLayer
  models/losses.py       # PDE residual, BC loss, data loss
```

**Website integration:** Keep existing LBM slider (contour | streamline). Add below it a 3-panel static comparison (C++ LBM / PINN surrogate / absolute error delta). Later add a live ONNX Runtime Web inference page (PINN-only; no C++ WASM rebuild) with a frame slider and field selector.

#### Phase 6.6: Interactive Flow Viewer (Canvas Engine)

**Goal:** Replace the static LBM/PINN "parametric interpolation" section with an
interactive, animated flow viewer on every case page. The viewer streams compact
binary frame data exported from the LBM solver and renders velocity-magnitude
contours with overlaid streamlines directly on a `<canvas>` (no images, no WASM
C++ rebuild).

**Files created:**
```
docs/assets/js/colormaps.js      # viridis/turbo/jet/rdbu LUTs, paintField() (flipY-aware)
docs/assets/js/flow-viewer.js     # FlowViewer class: LBM time-series canvas engine
docs/assets/js/pinn-inference.js  # PinnSurrogate: precomputed sweep + ONNX upgrade
docs/assets/js/vendor/onnxruntime-web/dist/  # vendored ort.min.js + wasm (numThreads=1)
pinn/export/export_web_data.py    # LBM frames -> float16 .bin (gz), PINN sweep, ONNX model
docs/assets/data/{case}/          # lbm_re{val}.bin (+ gz), pinn_*.bin, pinn_model.onnx
```

**Binary format** (`pinn/export/export_web_data.py`): header magic `0x4C424D31` + uint32
n_frames, nx, ny, n_chan, dtype_flag, then float16 (or float32) little-endian data
in `[frame][channel][y][x]` order. JS parser at `window.FlowData.parseBinary()`.

**Viewer layout (per case page):** Two SEPARATE sections (not tabs):
1. **LBM Evolution** -- canvas + Play/Pause + scrubber + frame counter. Auto-plays
   the solver's 51-200 frames from rest to steady state.
2. **PINN Prediction** -- canvas + discrete Re buttons (100/400/1000) + time
   scrubber + inference-time readout. Uses the precomputed PINN temporal binaries
   (`pinn_temporal_re{re}.bin`) by default; upgrades to live ONNX Runtime Web
   inference of the trained ParametricPINN when `pinn_temporal_model.onnx` exists.

**Orientation note:** LBM frame data is stored y-up (y=0 = bottom wall, y=NY-1 =
lid/top). `paintField()` uses `flipY=true` so the lid renders at the canvas TOP.
Streamlines must be drawn with the same transform: `canvas_y = ny - 1 - data_y`.

**Current bug (to fix):** Streamlines in `flow-viewer.js` + `pinn-inference.js` are
drawn in raw data-y (canvas top = bottom wall), so they appear flipped relative to
the contour. Fix: transform y-coordinates in `ctx.moveTo/lineTo`.

**Colormap + layout convention (new):**
- Per-case colormap: every plot AND animation on a case uses ONE colormap
  for visual consistency. Static PNGs get it from `postprocess.py` `CASE_CMAPS`
  (keyed by shape); the animation gets it from `FlowViewer({ cmap })` in the
  page init. Cavity = `viridis`; Backward-Step = `coolwarm` (another, so
  cases read differently at a glance).
- `FlowViewer` now takes a `cmap` option (default `viridis`) used by both
  `_render` (contour) and `_drawStreamlines` (speed-colored streamlines).
- In `cavity.html` the LBM Evolution animation was moved INTO the
  **Velocity Field** section, directly below the static contour/streamline
  comparison slider, so the reader sees the steady field first, then watches
  the flow develop. The PINN Prediction (temporal) section stays separate.
- Static slider + animation share footprint: `.comparison-slider` is now
  `aspect-ratio: 1/1; max-width: 560px` (same as `.fv-stage`) and its
  images use `object-fit: cover`, so both render square at the same width. The
  cavity slider PNGs were regenerated square (`scripts/gen_cavity_slider.py`).

**Status:** In progress. Cavity page has the 5-tab viewer (LBM/Compare/Particles/Error/
Live PINN); simplify to the two-section layout and velocity-only, fix orientation,
then replicate on all case pages.

#### Phase 6.8: Time-Parametric PINN (Spatio-Temporal Surrogate)

**Goal:** Extend the parametric PINN to learn the FULL transient evolution, not just
steady state. This is the highest-impact ML deliverable: a single network predicts
`(u, v, p)` at any `(x, y, Re, t)`, enabling true ML-powered animation that the LBM
section cannot match in interactivity.

**Architecture:** Input `(x, y, Re_n, t_n)` where `t_n = frame_index/(n_frames-1)`
is normalized simulation time. Fourier features on (x, y) only -> 512-dim, then
concatenate `Re_n` and `t_n` -> 514-dim MLP input. MLP: 256 hidden, 8 layers, tanh
(~600K params). Output `(u, v, p)`.

**Unsteady PDE residual:** Add material time derivative to the steady NS used in 6.3:
```
du/dt + u·du/dx + v·du/dy = -dp/dx + nu*(d2u/dx2 + d2u/dy2)
dv/dt + u·dv/dx + v·dv/dy = -dp/dy + nu*(d2v/dx2 + d2v/dy2)
du/dx + dv/dy = 0
```
`du/dt`, `dv/dt` via torch.autograd on the `t` input.

**Training data:** 51-frame LBM sequences at Re=100 and Re=400 (importance-sampled
sensors, 3000/frame). Hybrid loss = w_pde*unsteady_NS + w_data*data(u,v,p) + w_bc*BC.

**Files to create:**
```
pinn/data/temporal_loader.py   # Load LBM frames -> (x,y,Re_n,t_n) -> (u,v,p) samples
pinn/cases/cavity/train_temporal.py  # Multi-Re temporal training loop (MPS)
pinn/cases/cavity/export_temporal.py # ONNX + precomputed temporal sweep binaries
```

**Web integration:** Add a time scrubber (or auto-play) to the PINN Prediction
section. The surrogate then animates flow evolution frame-by-frame, matching the
LBM Evolution section for direct solver-vs-ML comparison.

**Implementation status (done):**
- `pinn/data/temporal_loader.py`: loads 51-frame LBM sequences at Re=100/400,
  emits importance-sampled `(x,y,Re_n,t_n) -> (u,v,p)` sensors + collocation.
- `pinn/models/losses.py`: added `unsteady_pde_loss_multi_re`,
  `bc_loss_cavity_temporal`, `ic_loss_cavity_temporal`, `total_loss_cavity_temporal`
  (IC enforces rest state at t_n=0 across Re).
- `pinn/cases/cavity/train_temporal.py`: multi-Re temporal training loop (Adam + L-BFGS),
  `ParametricPINN(n_params=2)` (Fourier on x,y, then Re_n + t_n -> 514-dim input).
- `pinn/cases/cavity/export_temporal.py`: writes `pinn_temporal_re{re}.bin` (+ `.gz`) per-Re
  frame sequences (same binary format as LBM viewer) + `pinn_temporal_model.onnx`.
- `docs/assets/js/flow-viewer.js`: added `filePrefix` option (loads
  `pinn_temporal_re{re}.bin`) + graceful `_showMissing()` placeholder when the
  surrogate binary is absent.
- `docs/cavity.html`: PINN Prediction section now drives a second FlowViewer over
  the temporal binary, with its own Re buttons + play/pause + time scrubber
  (`pinnReGroup`, `pinnPlay`, `pinnScrubber`, `pinnFrameLabel`).

**Training results (temporal, done):**
- 12,000 Adam + 1,000 L-BFGS epochs, 201 min on MPS, 593,155 params
- Final loss 1.2e-3 (pde 3.4e-5, data 1.4e-4, bc 3.5e-5, ic 3.6e-5)
- Frame-by-frame L2: re100 u mean=33.3% final=29.9%, v mean=48.0%;
  re400 u mean=33.0% final=34.7%, v mean=43.1%. Early transient (frames 0-10) hardest at ~45% L2.
- u_max ratio 1.13-1.16 (vortex strength captured). Supersedes steady-state model.
- Output: `output/cavity/pinn_temporal/model_temporal.pt`,
  `docs/assets/data/cavity/pinn_temporal_re{100,400}.bin` (+.gz),
  `docs/assets/data/cavity/pinn_temporal_model.onnx` (2.38 MB)

**Status:** Completed. Re=1000 extension (Phase 6.8b): data exists
(`output/cavity/re1000/frames/`, 51 frames at 256x256), retrain pending.

#### Phase 6.9: Model Improvement Roadmap

Prioritized enhancements to raise surrogate accuracy and design-space coverage.
Ordered by impact-to-effort ratio.

| # | Improvement | Problem it fixes | Approach | Priority |
|---|-------------|------------------|----------|----------|
| 1 | Pressure-Poisson residual | Pressure prediction near-constant (std 0.0015 vs true 0.0413) | Add Laplacian(p) = -rho*(d2(uu)/dx2 + 2*d2(uv)/dxdy + d2(vv)/dy2) to hybrid loss | High |
| 2 | Extended Re range (100-1000) | Only Re=100/400 trained; narrow design space | Add Re=1000 data (exists at 256x256), retrain 3-Re temporal model | High |
| 3 | Curriculum learning | Early transient (frames 0-10) hardest at ~45% L2 | Train on early frames first, then expand time window; time-weighted data loss | Medium |
| 4 | Adaptive importance sampling | Fixed 3000 sensors under-resolve moving vortex core | Resample sensors each epoch weighted by current residual/gradient | Medium |
| 5 | Multi-scale Fourier features | Single sigma=5.0 mis-resolves boundary layers vs bulk | Concatenate multiple sigma bands (e.g. 1, 5, 20) into feature layer | Medium |
| 6 | Temporal attention / recurrent | MLP treats t_n as static input, weak time coupling | Add a lightweight temporal attention or GRU head over t_n | Low |
| 7 | Ensemble training (UQ) | No uncertainty estimate on predictions | Train N models with different seeds; report mean + std as error band | Low |

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
    cases/cavity/              # train_steady.py, train_temporal.py, export_sweep.py,
                               #   export_temporal.py, plot_results.py, logs/
    cases/cylinder/            # train.py, evaluate.py
    export/export_web_data.py  # LBM frames -> float16 .bin (+.gz) for website
    data/loader.py             # Read frame*.json + forces.jsonl -> numpy arrays
    data/temporal_loader.py    # Load LBM frames -> (x,y,Re_n,t_n) -> (u,v,p) samples
    models/pinn.py             # PINN + ParametricPINN MLP architectures
    models/losses.py           # PDE residual, BC loss (cavity/cylinder), data loss

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

## Solver Accuracy & Capability Upgrade Roadmap (2026-07-18)

**Context:** Current solver uses Cartesian immersed boundary + Bouzidi interpolated
bounce-back, no near-wall treatment (pure no-slip), isothermal incompressible only,
and is strictly 2D (D2Q9 hardcoded). The goal is to improve accuracy (smooth curved
boundaries, proper y+ / wall treatment), add pressure visualization, thermal physics,
and a 3D architecture.

### UPGRADE 1: Curved Boundary Accuracy (blocky -> smooth)

Three tiers, easiest first:

- **Tier 1 (immediate): Grid refinement.** Increase grid from 800x300 to 2400x900 for
  curved cases. At radius=90 cells (vs 30), staircase error drops 3x. Bouzidi `q`
  already gives sub-grid positioning. Cost: 9x memory/compute (feasible 2D on M5).
  **Status: Available via --nx/--ny override (main.cpp).**
- **Tier 2 (medium effort, recommend first): Filippova-Hanel / Mei interpolated
  bounce-back.** Bouzidi is 2nd-order but unstable at q->0 or q->1. Replace with the
  Mei et al. (1999) formula in `apply_bouzidi_bb()` (lbm.hpp:206-245):
  `f_bb = q*f_i^{eq}(rho,u_wall) + (1-q)*f_post + (2q-1)*w_i*rho*(e_i.u_wall)/cs^2`.
  Unconditionally stable for all q in [0,1]. Implemented as `BounceBackGeometry::use_mei_bb`
  (default true). **Status: Completed (2026-07-18), Cd~1.53 at Re=100 (vs 1.77 Bouzidi).**
- **Tier 3 (high effort): IBM with direct forcing.** For airfoils/complex shapes. New
  `src/ibm.hpp` (~200 lines). Lagrangian points on true surface (reuse `naca_coords()`),
  force spreading via 4-point smoothed delta, velocity interpolation. Add only for
  airfoil case. **Status: In progress (2026-07-18).**

### UPGRADE 2: Wall Functions / y+ Requirements

- **Wall-distance computation:** New `compute_wall_distance()` in lbm_types.hpp -- BFS from
  obstacle nodes, returns distance in lattice units. **Status: Completed (2026-07-18).**
- **Van Driest damping for LES:** Implemented in `mrt_collide()` (lbm.hpp:165-179):
  `nu_t_damped = nu_t * (1 - exp(-y+/A+))^2` with A+ = 26. Prevents over-damping near walls.
  **Status: Completed (2026-07-18), validated Cd~1.77 at Re=100.**
- **Wall function bounce-back (WFB):** Slip-velocity approach (Ponsin et al. 2025).
  Compute wall shear stress from resolved gradient, use log-law `u_slip = u_tau*(1/kappa*ln(y+)+B)`,
  impose via modified bounce-back. New `src/wall_functions.hpp` (~150 lines).
  **Status: In progress (2026-07-18).**
- **y+ in lattice units:** `y+ = y*u_tau/nu` where `u_tau = sqrt(tau_wall/rho)` and
  `nu = (tau-0.5)/3`. For channel at Re_tau=180 with NY=200, first cell y+ = 0.9 (resolved).
  At NY=30: y+ = 6 (buffer layer, wall function needed).

### UPGRADE 3: Pressure Contours & Enhanced Vorticity

- **Pressure Cp plot:** Added `--cp` flag in postprocess.py, `save_cp_png()` function.
  `Cp = (p - p_ref)/(0.5*rho_inf*U_inf^2)`. **Status: Completed (2026-07-18).**
- **Pressure channel in viewer:** Binary format already supports multiple channels.
  Add field selector (velocity / pressure / vorticity / temperature) to flow-viewer.js.
  **Status: Pending (needs front-end work).**
- **Full-resolution vorticity:** Upgraded to 9-point stencil in lbm.hpp:542-558.
  **Status: Completed (2026-07-18), higher-order accuracy.**

### UPGRADE 4: Thermal LBM (Heat Transfer)

- **Approach: Double Distribution Function (DDF).** Keep D2Q9 `f_i` for momentum, add
  D2Q9 (or D2Q5) `g_i` for temperature. Solves `dT/dt + u.grad(T) = alpha*laplacian(T)`.
- **Boussinesq coupling:** `F_buoyancy = -rho_0*beta*(T-T_ref)*g` added to momentum.
- **Parameters:** Pr = nu/alpha (0.71 air), Ra = g*beta*dT*L^3/(nu*alpha), Nu = h*L/k.
- **New file: `src/thermal.hpp`** (~200 lines): `g_i` collision, streaming, thermal BCs
  (isothermal Dirichlet, heat flux Neumann, adiabatic bounce-back).
- **Modify `lbm_types.hpp`:** Add `g_thermal`, `g_thermal_next` to LBMCapabilities.
- **New entry point: `src/heated_cylinder.cpp`** (Nusselt validation) + natural
  convection cavity (Ra=10^3 to 10^6 benchmark).
- **Memory overhead:** 2x (`f` + `g` vectors). For 800x300: ~16 MB extra. Negligible.
- **Status: In progress (2026-07-18).**

### UPGRADE 5: 3D LBM Architecture

- **Phase 5a: D3Q19 lattice abstraction.** New `src/lattice.hpp` with D2Q9/D3Q19 structs
  (cx, cy, cz, weights). Template `LBMCapabilities<Lattice>` on lattice type. NZ=1 with
  D2Q9 is a degenerate D3Q19 (all z-velocities zero).
- **Phase 5b: Performance.** LBM is memory-bandwidth bound. On M5: 200^3 D3Q19 = ~1.2 GB,
  fits in unified memory. OpenMP 3D decomposition natural. ~50-100 MLUPS expected.
  **Use D3Q19** (19 dirs, 152 B/node) not D3Q27 (27 dirs, 216 B/node) for engineering
  accuracy. D3Q27 only for DNS of isotropic turbulence.
- **Phase 5c: Incremental migration.** (1) Abstract lattice into lattice.hpp. (2) Template
  core solver on Lattice. (3) Keep 2D default (NZ=1). (4) Test D3Q19 with NZ=1 (verify
  identical to D2Q9). (5) Enable NZ>1 for true 3D. Avoids complete rewrite.
- **Phase 5d: New 3D entry points:** `main_3d.cpp` (sphere), `cavity_3d.cpp`, `pipe_3d.cpp`.
- **Phase 5e: 3D postprocessing:** Slice planes (xy/xz/yz) in FlowViewer, isosurface
  rendering (Q-criterion), VTK output for ParaView (already partial for 2D).
- **D3Q19 MRT:** 19 moments, new transformation matrices (d'Humieres 2002). More relaxation
  rates than D2Q9.

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

## Known Issues

1. **MRT parameter tuning**: Relaxation rates s_e, s_eps, s_q, s_pi need tuning per Re for optimal stability. Defaults for Re=100 may not extend to Re=1000 without adjustment (mitigated by LES).
2. **Polygon Bouzidi**: q_polygon() detects intersection but cannot distinguish inside/outside the polygon; works for convex polygons where the ray exits through one edge. Concave polygon or complex shapes may produce incorrect q.
3. **2D Ahmed body**: The real Ahmed body flow is 3D with strong longitudinal vortices at the slant. A 2D slice captures only the base pressure drag trend, not the characteristic Cd drop at 25 deg.
4. **AMR coarse-fine interface**: Prolongation/restriction operators may produce spurious reflections at block boundaries. Mitigated by ghost cell overlap and validation against single-grid baseline.
5. **LES at very low Re**: Smagorinsky adds negligible eddy viscosity at Re < 200. Should match non-LES baseline within 0.1%.
6. **forces.jsonl**: Static cache approach assumes single output directory per process. Safe for all batch scripts (each is a separate process).
7. **Curved boundary accuracy**: Bouzidi bounce-back is 2nd-order but has stability issues at extreme q. Blocky staircase for coarse grids. See Upgrade 1 (Tier 2 Mei bounce-back recommended).
8. **No near-wall treatment**: Pure no-slip bounce-back everywhere. No y+ computation, no wall function. See Upgrade 2.
9. **Isothermal only**: No thermal modeling. See Upgrade 4.
10. **Strictly 2D**: D2Q9 hardcoded. No 3D capability. See Upgrade 5.

## Implementation Sequence

1. **Phase 1: Smagorinsky LES**, Completed
2. **Phase 2: Block-Structured AMR**, In progress (restriction operator needs fix)
3. **Phase 3: Vorticity + Postprocessor**, Completed
4. **Phase 4: Full Re-Runs + New Cases**, In progress (17 simulations pending)
5. **Phase 5: Website Updates**, In progress (interactive LBM/PINN viewers on all cases)
6. **Phase 6.8: Time-Parametric PINN**, Completed (Re=100/400/1000 trained, ONNX + binary export done)
7. **Phase 6.8b: Re=1000 temporal extension**, Completed (trained, exported; Re=300 interpolation panel added)
8. **Phase 6.9: Model improvement roadmap**, Pending (pressure-Poisson, Re range, curriculum)
9. **Phase 5.5: Cavity deep dive + PINN narrative**, Completed (website restructuring template)

#### Phase 5.5: Cavity Page Deep Dive + PINN Surrogate Narrative

**Goal:** Restructure the cavity case page into a portfolio-grade analysis: scanner-friendly Key Findings, a real LBM Analysis section, expanded PINN narrative (architecture, training, steady-state, temporal, parametric interpolation), a Training Convergence plot, a "What the PINN Unlocks" applications section, and an honest Limitations & Next Steps section. This establishes the template for all other case pages.

**Completed (website restructuring only; no new simulations):**
- `docs/cavity.html`: restructured to Hero -> Setup -> Velocity Field -> Validation -> Key Findings -> LBM Analysis -> PINN (Architecture, Training, Steady-State Comparison x2, Temporal Surrogate, Accuracy Summary, Parametric Interpolation) -> Training Convergence -> What the PINN Unlocks (6 subsections) -> Limitations & Next Steps -> Footer
- `pinn/cases/cavity/plot_loss_convergence.py`: training loss convergence plot (`loss_convergence.png`), total loss 1.26Mx reduction
- `pinn/cases/cavity/plot_temporal_l2.py`: frame-by-frame temporal L2 profile (`temporal_l2_profile.png`), Re=100 u-mean 33.6% / v-mean 48.6%, Re=400 u-mean 33.6% / v-mean 44.5%
- CSS: `.key-findings` compact bullet list style in `docs/css/style.css`
- Speed section (~97 ms/surrogate frame vs ~30 s LBM, ~300-600x), Re=300 interpolation panel, sensitivity map (`sensitivity_map_re300.png`) -- all done.

**Deferred to future roadmap (simulations):**
- Phase 6.4 backward-facing step PINN (data re-run needed)
- Phase 6.5 orifice plate parametric PINN (new Re+geometry sweeps needed)
- Phase 6.7 ablation study (data/PDE/BC terms)
- Phase 6.3d pressure-Poisson coupling (fix near-constant pressure)

**Carry-over template (per case page):**
1. Hero + Setup table
2. Velocity Field (side-by-side: steady slider | LBM evolution)
3. Validation (quantitative comparison)
4. Key Findings (3-4 bullets)
5. LBM Analysis (flow physics)
6. PINN Surrogate (architecture, training, steady-state, temporal, parametric)
7. Training Convergence (loss plot)
8. What the PINN Unlocks (applications)
9. Limitations & Next Steps

Infrastructure already generalized: `pinn/export/export_web_data.py` CASES dict, `pinn/data/temporal_loader.py` multi-Re loading, `pinn/cases/cavity/train_temporal.py` multi-Re training, `docs/assets/js/flow-viewer.js` filePrefix + cmap options.

## Active Simulation Checklist (post-solver-upgrade)

The solver changed significantly: Mei/Filippova-Hanel bounce-back (default on),
Van Driest LES damping, thermal DDF, IBM, wall functions. All existing cases must
be re-run to confirm validation still holds, and new thermal/IBM cases must be
produced. Check off as completed. Use `--use-les` where tau < 0.55.

### A. Re-validate existing cases with Mei bounce-back (Cd/St vs literature)

- [ ] **Cylinder Re=100** -- `./build/LBM_Engine 100` (expect Cd~1.53, was 1.77 Bouzidi)
- [ ] **Cylinder Re=200** -- `./build/LBM_Engine 200` (expect Cd~1.37, Cl amp~0.37)
- [ ] **Square cylinder Re=200** -- `./build/LBM_SquareCylinder 200` (Cd~1.16, ERCOFTAC)
- [ ] **Flat plate AoA=0 Re=1000** -- `./build/LBM_FlatPlate 1000 0` (Cd~0.026, Blasius)
- [ ] **Flat plate AoA=10 Re=1000** -- `./build/LBM_FlatPlate 1000 10` (Cl~1.1)
- [ ] **Cavity Re=100** -- `./build/LBM_Cavity 100` (Ghia u-profile)
- [ ] **Cavity Re=400** -- `./build/LBM_Cavity 400`
- [ ] **Step Re=100** -- `./build/LBM_Step 100` (Armaly Xr/H)
- [ ] **Orifice 1p1h Re=100** -- `./build/LBM_OrificePlate 100 1p1h` (ISO 5167 K)
- [ ] **Periodic hills Re=100** -- `./build/LBM_PeriodicHills 100` (LES benchmark)
- [ ] **Cylinder near wall Re=100 gap=20** -- `./build/LBM_CylinderNearWall 100 20`
- [ ] **Side-by-side Re=100 S/D=3** -- `./build/LBM_SideBySide 100 3`
- [ ] **Rotating cylinder Re=100 w=1.0** -- `./build/LBM_RotatingCylinder 100 1.0`
- [ ] **Urban canyon side AR=0.5** -- `./build/LBM_UrbanCanyon --mode side --ar 0.5`
- [ ] **Downwash Re=100** -- `./build/LBM_Downwash 100`

### B. Van Driest damping sanity (LES cases, confirm no regression)

- [ ] **Cylinder Re=1000 fine grid (NX=2400 NY=900)** -- `./build/LBM_Engine 1000 40000 --use-les` (stable, Cd within 30% lit)
- [ ] **Periodic hills Re=1000** -- `./build/LBM_PeriodicHills 1000` (LES, check y+ damping)
- [ ] **Urban topdown horizontal Re=100 (LES)** -- `./build/LBM_UrbanCanyon --mode topdown --horizontal --use-les`

### C. New thermal LBM cases (Upgrade 4)

- [ ] **Heated cylinder Re=100 T_wall=1.5** -- `./build/LBM_HeatedCylinder 100 1.5 40000` (Nu vs Eshghy 1970)
- [ ] **Heated cylinder Re=200 T_wall=1.5** -- `./build/LBM_HeatedCylinder 200 1.5 40000`
- [ ] **Heated cylinder + buoyancy Re=100** -- `./build/LBM_HeatedCylinder 100 1.5 40000 --buoyancy` (mixed conv)
- [x] **Natural convection cavity Ra=1e4** -- `./build/LBM_ThermalCavity 1e4` (entry point created, runs stable; Nu~5.3 vs lit 2.24 -- BC/stencil needs refinement, see note G)
- [ ] **Natural convection cavity Ra=1e5** -- Ra sweep for benchmark
- [ ] **Natural convection cavity Ra=1e6** -- highest Ra (D2Q9 MRT stability check)

### D. New IBM cases (Upgrade 1 Tier 3)

- [ ] **NACA 2412 AoA=0 Re=1000** -- `./build/LBM_AirfoilIBM 1000 0 40000` (Cl~0, Cd~0.01)
- [ ] **NACA 2412 AoA=5 Re=1000** -- `./build/LBM_AirfoilIBM 1000 5 40000` (Cl~0.55, thin-airfoil)
- [ ] **NACA 2412 AoA=10 Re=1000** -- `./build/LBM_AirfoilIBM 1000 10 40000` (Cl~1.1)
- [ ] **NACA 0012 AoA=0 Re=500** -- symmetric, baseline
- [ ] **NACA 0012 AoA=8 Re=500** -- symmetric, Cl~0.88 (validation)

### E. Wall function validation (Upgrade 2, after WFB integration)

- [ ] **Channel flow Re_tau=180** -- NEW entry point `LBM_Channel` (log-law profile)
- [ ] **Flat plate turbulent Re=1e5** -- `LBM_FlatPlate 100000 0 --use-wf` (Cf vs Blasius)
- [ ] **Cylinder Re=1000 coarse grid (NX=400)** -- WFB vs resolved-y+ compare

### F. Post-processing / visualization updates

- [ ] **Cp contours** for cylinder Re=100: `python3 scripts/postprocess.py output/cylinder/re100 --cp`
- [ ] **Temperature contours** for heated cylinder: `scripts/postprocess.py output/heated_cylinder_re100_tw150 --field temperature` (add temperature field key)
- [ ] **Regenerate all case PNGs** with new Mei BB (existing outputs stale)
- [ ] **Update docs/** case pages with new Cd values (Mei BB shifted Cd ~15% lower)
- [ ] **Add heated_cylinder.html** + airfoil_ibm.html case pages

### G. Notes

- Mei BB lowers Cd vs Bouzidi (~1.53 vs 1.77 at Re=100). Update validation tables
  to reflect the new (more accurate) values; keep Bouzidi result as "legacy".
- Thermal Nu on coarse grid reads high (~145 vs lit ~5-10 at Re=100); finer grid
  or grid-convergence study needed for publication-grade numbers.
- IBM airfoil uses Eulerian obstacle mask for streaming skip + Lagrangian force
  for no-slip; confirm Cl slope matches 2*pi*alpha at small AoA.
- WFB (wall_functions.hpp) is infrastructure-only; integrate into streaming loop
  (`apply_wall_function_bb`) before running Section E.
- Natural convection cavity (LBM_ThermalCavity) runs stable but Nu_avg~5.3 at
  Ra=1e4 vs benchmark 2.24 (de Vahl Davis 1983): the equilibrium-Dirichlet thermal
  BC + 1-cell gradient stencil overestimates the wall heat flux. Needs either a
  finer grid (L>300) or a proper ghost-node/central-difference Nu stencil. Flow
  field (plume + recirculation) develops correctly; magnitude is the open issue.
