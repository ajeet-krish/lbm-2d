# Project Context -- LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

## Style Rules
- **No em dashes** in any file. Use two regular hyphens (--) instead of ---, &mdash;, &ndash;, or literal Unicode em dash.
- **C++ code style**: 4-space indentation, K&R braces, no tabs, no trailing whitespace.
- **Variable naming**: snake_case for local variables, PascalCase for structs/classes, SCREAMING_SNAKE_CASE for constants.
- **HTML/CSS**: Double quotes for attributes, 2-space indentation, semantic HTML5 elements.

## Goal
Build and deploy a cache-optimized D2Q9 Lattice Boltzmann Method CFD solver in C++20 as a portfolio centrepiece for aerospace/defense engineering roles. Deliver a 5-page HTML portfolio with an interactive canvas animation player, KaTeX theory equations, and a production-grade GitHub repository with CI and unit tests.

## Target Audience
Aerospace hiring managers at SpaceX, Firefly Aerospace, Lockheed Martin, Blue Origin, and similar. The site must communicate: HPC competency (C++, OpenMP, cache optimization), CFD fundamentals (validation against known benchmarks), and engineering communication skills (interactive web presentation).

## Current Status (2026-07-09)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project scaffolding (CMake, directories, Git) | ✅ |
| 2 | Core solver (D2Q9, BGK, collide+stream, boundaries) | ✅ |
| 3 | Force extraction (momentum exchange, Cd/Cl/St) | ✅ |
| 4 | OpenMP parallelization (collapse(2)) | ✅ |
| 5 | Google Test suite (unit + integration tests) | ✅ |
| 6 | GitHub Actions CI | ✅ |
| 7 | VTK export + batch runner | ✅ |
| 8 | Website CSS theme (CFD Jet) | ✅ |
| 9 | index.html (home page) | ✅ |
| 10 | simulation.html (interactive canvas player) | ✅ |
| 11 | theory.html (LBM equations with KaTeX) | ✅ |
| 12 | implementation.html (code architecture) | ✅ |
| 13 | results.html (validation + field galleries) | ✅ |
| 14 | scripts/postprocess.py (VTK -> JSON/PNG) | ✅ |
| 15 | docs/assets/js/viewer.js (interactive viewer) | ✅ |
| 16 | Validate Strouhal vs Williamson for Re=20-200 | ✅ |
| 17 | Generate simulation frames + validation plots | ✅ |
| 18 | Lid-driven cavity solver | ✅ |
| 1C | NACA 4-digit airfoil solver + polygon obstacle placement | ✅ |
| 19 | Expanded cylinder Re values (60, 80, 120, 160, 180) | ❌ Cancelled |
| 20 | Airfoil results page (Cl/Cd plots, field gallery) | ✅ |
| 21 | WebAssembly real-time simulation (Emscripten) | ⏳ |
| 22 | Final deploy to GitHub Pages | ⏳ |

## Roadmap

### Phase 1: Expanded Simulation Set

| Item | Description | Effort |
|------|-------------|--------|
| 1A | Lid-driven cavity -- square domain, 3 stationary walls + moving lid. Non-equilibrium bounce-back for top boundary. Validate against Ghia 1982 centerline velocity profiles. Re=100, 400, 1000. | 2-3h |
| 1B | Expanded cylinder Re: add 60, 80, 120, 160, 180 to fill validation curve gaps. No code changes, just re-run. | 30min |

**Code changes for 1A:**
- `lbm_types.hpp`: Make NX/NY runtime variables (remove constexpr). Add `CaseType` enum.
- `lbm.hpp`: Add `place_walls()` for cavity boundaries. Add `enforce_lid(U_lid)` with momentum-corrected bounce-back. Skip periodic BC and inflow/outflow in cavity mode.
- `cavity.cpp`: New entry point (./build/LBM_Cavity <Re> <nx> <steps>).

**Code changes for 1C (NACA airfoil):**
- `geometry.hpp`: New file -- NACA 4-digit coordinate generation, 2D transform, point-in-polygon.
- `lbm.hpp`: Add `place_polygon()` for arbitrary polygon obstacles.
- `airfoil.cpp`: New entry point (./build/LBM_Airfoil <series> <Re> <AoA> <steps>).
- `CMakeLists.txt`: LBM_Airfoil executable target added.
- `scripts/plot_airfoil.py`: Validation plots (Cl vs AoA, Cd vs AoA, drag polar).

### Phase 2: Expanded Results Page

| Item | Description | Effort |
|------|-------------|--------|
| 2D | Lid-driven cavity section: centerline u-velocity profiles vs Ghia 1982, vorticity contours. | Done in Phase 1A |
| 2E | Merge simulation.html into results.html as embedded canvas viewer tab. | 30min |

### Phase 3: WebAssembly Real-Time Solver

Replaces `simulation.html` (pre-computed playback) with a live WASM-powered simulator on a 100x60 coarse grid. User selects shape (cylinder, square, diamond, star, NACA airfoils with AoA slider) and Re (20-200), sees the flow evolve organically with animated PhiFlow-style pathline streaks.

| Item | Description | Effort | Status |
|------|-------------|--------|--------|
| 3A | Emscripten toolchain + build script (`build_wasm.sh`) | 15 min | ⏳ |
| 3B | WASM solver port (`wasm_main.cpp`) -- no OpenMP, no file I/O, exported C functions: `wasm_init`, `wasm_step`, `wasm_set_shape`, `wasm_get_vel_ptr`, `wasm_get_cd/cl` | 2-3h | ⏳ |
| 3C | Canvas rendering engine (`wasm_sim.js`) -- velocity colormap, 200 particle pathlines with bilinear interpolation, obstacle outline, adaptive steps-per-frame | 3-4h | ⏳ |
| 3D | Interactive page (`simulation.html`) -- shape selector, Re slider, AoA slider (airfoil only), speed control, play/pause, Cd/Cl HUD | 2-3h | ⏳ |
| 3E | Shape library + initial state + polish | 1h | ⏳ |

**Architecture:**
```
User Input --> JS Controls --> WASM Solver (100x60 grid)
                                   |
                                   | step() called ~20-30x per frame
                                   | velocity field in HEAPF64 shared memory
                                   v
                              requestAnimationFrame loop
                                   |
                                   |--- Velocity jet colormap to canvas 2D
                                   |--- Pathline particle advection
                                   |--- Obstacle outline overlay
                                   |--- Cd/Cl HUD
```

**Key design decisions:**
- 100x60 grid (6K nodes) for interactive rates (~20-30fps on modern hardware)
- WASM solver runs ~20-30 steps per render frame (adaptive), flow develops over ~15-30s
- Pathlines: ~200 particles seeded upstream, bilinear velocity interpolation, fading trails
- All shapes via polygon obstacle: `place_polygon()` with generated vertex lists
- `simulation.html` replaced with WASM version; pre-computed viewer removed
- No initial state pre-computation -- start from rest, let flow develop organically

## File Layout

```
lbm-2d/
  AGENTS.md                    # This file
  README.md                    # Project overview
  CMakeLists.txt               # Build system (OpenMP + Google Test + Emscripten)
  .gitignore

  src/
    lbm_types.hpp              # D2Q9 constants, structs, index helpers
    lbm.hpp                    # Core solver: collide, stream, BC, force extraction
    geometry.hpp               # NACA 4-digit coords, polygon ops, point-in-polygon
    main.cpp                   # Cylinder flow entry point
    cavity.cpp                 # Lid-driven cavity entry point
    airfoil.cpp                # NACA 4-digit airfoil analysis entry point
    wasm_main.cpp              # WASM entry point (Phase 3)
    lbm_test.cpp               # Google Test suite (14+ tests)

  scripts/
    postprocess.py             # VTK -> JSON/PNG conversion for web viewer
    plot_airfoil.py            # Airfoil validation plots (Cl/Cd vs AoA, drag polar)
    run_all_re.sh              # Batch runner: builds + runs Re sweep
    run_cavity.sh              # Batch runner: lid-driven cavity sweep
    run_airfoil.sh             # Batch runner: airfoil AoA sweep
    build_wasm.sh              # Emscripten WASM build script (Phase 3)

  .github/workflows/
    ci.yml                     # GitHub Actions: build + test on ubuntu + macos

  docs/
    index.html                 # Home: hero, teaser video, results stats
    simulation.html            # WASM real-time solver (Phase 3 -- replaces pre-computed)
    theory.html                # LBM theory with KaTeX equations
    implementation.html        # Code architecture + source blocks
    results.html               # Validation plots + field galleries + cavity + airfoil
    css/
      style.css                # CFD Jet theme (dark, cyan/orange accents)
    assets/
      js/
        viewer.js              # Pre-computed canvas animation player (archive)
        wasm_sim.js            # WASM real-time rendering engine (Phase 3)
      data/                    # Pre-computed JSON (51 frames per Re, committed)
        cavity/                # Cavity JSON data
      images/                  # Field renders + validation plot PNGs (committed)
      videos/                  # MP4 animations (gitignored)

  output/                      # Simulation VTK frames (gitignored)
    re100/                     # Per-Re subdirectories
    cavity/                    # Cavity subdirectories (Phase 1)
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
Re = u_inflow * D / nu                          (cylinder flow, D = cylinder diameter)
Re = u_lid * NX / nu                            (lid-driven cavity, NX = cavity width)
tau = 0.5 + 3 * u_inflow * D / Re
tau = 0.5 + 3 * u_lid * NX / Re                (cavity version)

### Boundary conditions
- Zou/He velocity inlet: enforce u = u_inflow at x=0, compute rho from known distributions
- Convective outlet: zero-gradient extrapolation at x=NX-1
- Bounce-back: reverse distribution direction at obstacle nodes
- Periodic: wrap y-coordinate (top connects to bottom)
- Moving wall (lid-driven cavity): momentum-corrected bounce-back at top lid:
  f_{-i} = f_i + 6 * w_i * rho * (e_i . u_lid) for directions streaming into the wall

### Momentum exchange method
Force on cylinder at each boundary link:
F_i = e_i * [f(fluid, i) - f(obstacle, bounce_back(i))]
Total force summed over all fluid-obstacle link pairs.
Cd = 2 * Fx / (rho * u^2 * D)
Cl = 2 * Fy / (rho * u^2 * D)

### OpenMP considerations
- Parallel regions use `#pragma omp parallel for collapse(2)` on nested (y, x) loops
- Collision and streaming are separate parallel regions (cannot fuse without race conditions on streaming)
- All data is in std::vector -- no false sharing on modern x86/ARM with 64B cache lines
- Thread count controlled by OMP_NUM_THREADS environment variable

### Validation targets (Williamson 1988, Tritton 1959)
| Re | Strouhal | Cd_mean | Notes |
|----|----------|---------|-------|
| 20 | --       | ~2.0    | Steady, symmetric |
| 40 | --       | ~1.5    | Steady, recirculating wake |
| 60 | ~0.14    | ~1.4    | Onset of shedding |
| 100| 0.164-0.172 | ~1.4 | Fully periodic |
| 200| 0.180-0.195 | ~1.3 | Periodic, 3D effects begin |

### Actual computed values (stair-step D=30, BGK)
| Re | Strouhal | Cd_mean | Cl_amplitude | Notes |
|----|----------|---------|--------------|-------|
| 20 | --       | 3.108   | 0.000        | Steady, ~55% over literature |
| 40 | --       | 2.293   | 0.000        | Steady, ~53% over literature |
| 100| 0.200    | 1.763   | 0.374        | Shedding, St ~16-22% high, Cd ~26% high |
| 200| 0.240    | 1.600   | 0.785        | Shedding, St ~23-33% high, Cd ~23% high |

### Grid sizing guide
- NX = 400, NY = 150 recommended (60K fluid nodes)
- Cylinder diameter D = NY/5 = 30 (30 grid points across cylinder -- good resolution)
- Cylinder center at x = NX/4 = 100, y = NY/2 + 1 = 76 (off-center to break symmetry)
- Domain extends 300 grid points downstream for wake development

## Color Scheme: CFD Jet Theme

```css
:root {
  --bg-primary:   #0d1117;   // near-black
  --bg-card:      #161b22;   // card background
  --bg-canvas:    #0a0e14;   // simulation area
  --border:       #21262d;   // subtle borders

  --cyan:         #00d4ff;   // primary accent (cold fluid)
  --cyan-dim:     #0099cc;
  --orange:       #ff6b35;   // secondary accent (warm fluid)
  --orange-dim:   #cc5529;
  --turquoise:    #00f5d4;   // highlight
  --green:        #39d353;   // terminal/success

  --fg:           #c9d1d9;   // primary text
  --fg-dim:       #8b949e;   // muted text
  --fg-muted:     #484f58;   // very muted
}
```

This mimics the Paraview jet colormap (blue -> cyan -> green -> yellow -> orange -> red)
which CFD engineers recognize immediately as the standard visualization color scheme.

# Reference Commands

```bash
cmake -B build && cmake --build build          # Build solver
./build/LBM_Engine                             # Run default (Re=100)
./build/LBM_Engine 200                         # Run Re=200
./build/LBM_Engine 100 12000                   # Run with custom steps
./build/LBM_Tests                              # Run unit tests

# Batch runner
bash scripts/run_all_re.sh                     # Runs Re = 20, 40, 100, 200
bash scripts/run_cavity.sh                     # Runs cavity Re = 100, 400, 1000
bash scripts/run_airfoil.sh                    # Runs airfoil AoA sweep

# Cavity simulation
./build/LBM_Cavity                             # Default (Re=100, 128x128)
./build/LBM_Cavity 400 128                     # Re=400 on 128x128
./build/LBM_Cavity 1000 256 50000             # Re=1000 on 256x256

# Airfoil simulation
./build/LBM_Airfoil 0012 1000 0               # NACA 0012 at Re=1000, AoA=0
./build/LBM_Airfoil 0012 1000 8 10000         # NACA 0012 at Re=1000, AoA=8, 10K steps
./build/LBM_Airfoil 2412 1000 4               # NACA 2412 at Re=1000, AoA=4

# Post-processing (requires Python + numpy + matplotlib)
python3 scripts/postprocess.py output/re100    # VTK -> PNG frames
python3 scripts/postprocess.py output/re100 --json  # VTK -> JSON for web
python3 scripts/plot_airfoil.py                # Generate airfoil validation plots

# WASM build (requires Emscripten)
bash scripts/build_wasm.sh                    # Build WASM solver
python3 -m http.server -d docs 8765           # Serve WASM page with correct MIME types

# Preview website
python3 -m http.server -d docs 8765
open http://localhost:8765

# Clean
rm -rf build output/re*

## CI Pipeline

`.github/workflows/ci.yml`:
- Trigger: push, pull_request to main
- Strategy matrix: ubuntu-latest, macos-latest
- Steps: checkout -> install OpenMP -> cmake configure -> cmake build -> run tests

## Known Issues

1. **VTK file size**: ASCII VTK files are ~200MB each for 400x150 grid. Use scripts/postprocess.py to downsample for web.
2. **Re > 300**: BGK becomes unstable at higher Re due to tau approaching 0.5. Use MRT or Smagorinsky LES for stability.
3. **Low Re (20, 40)**: Flow is steady. Increase steps to 20,000+ to reach steady state from rest initialization.
4. **Grid resolution**: 400x150 is moderate. For production CFD, 800x300+ is recommended with grid refinement near walls.
5. **Systematic Cd/St bias**: Stair-step cylinder boundary and BGK at moderate D=30 cause ~20-55% overprediction vs literature. Documented on results.html.
