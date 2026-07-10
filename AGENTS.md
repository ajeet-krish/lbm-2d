# Project Context -- LBM-2D: High-Performance C++ Lattice Boltzmann CFD Solver

## Style Rules
- **No em dashes** in any file. Use two regular hyphens (--) instead of ---, &mdash;, &ndash;, or literal Unicode em dash.
- **C++ code style**: 4-space indentation, K&R braces, no tabs, no trailing whitespace.
- **Variable naming**: snake_case for local variables, PascalCase for structs/classes, SCREAMING_SNAKE_CASE for constants.
- **HTML/CSS**: Double quotes for attributes, 2-space indentation, semantic HTML5 elements.

## Goal
Build and deploy a cache-optimized D2Q9 Lattice Boltzmann Method CFD solver in C++20 as a portfolio centrepiece for aerospace/defense engineering roles. Deliver a 5-page HTML portfolio with interactive Plotly dashboards, KaTeX theory equations, and a production-grade GitHub repository with CI and unit tests.

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
| 10 | simulation.html (interactive dashboard) | ✅ |
| 11 | theory.html (LBM equations with KaTeX) | ✅ |
| 12 | implementation.html (code architecture) | ✅ |
| 13 | results.html (validation + field galleries) | ✅ |
| 14 | scripts/postprocess.py (VTK -> JSON/PNG) | ⏳ |
| 15 | docs/assets/js/viewer.js (interactive viewer) | ⏳ |
| 16 | Validate Strouhal vs Williamson for Re=20-200 | ⏳ |
| 17 | Generate simulation video + validation plots | ⏳ |
| 18 | Final deploy to GitHub Pages | ⏳ |

## File Layout

```
lbm-2d/
  AGENTS.md                    # This file
  README.md                    # Project overview
  CMakeLists.txt               # Build system (OpenMP + Google Test)
  .gitignore

  src/
    lbm_types.hpp              # D2Q9 constants, structs, index helpers
    lbm.hpp                    # Core solver: collide, stream, BC, force extraction
    main.cpp                   # Simulation entry point + orchestration
    lbm_test.cpp               # Google Test suite (14+ tests)

  scripts/
    postprocess.py             # VTK -> JSON/PNG conversion for web viewer
    run_all_re.sh              # Batch runner: builds + runs Re sweep

  .github/workflows/
    ci.yml                     # GitHub Actions: build + test on ubuntu + macos

  docs/
    index.html                 # Home: hero, teaser video, results stats
    simulation.html            # Interactive dashboard: Re selector, Cd/Cl plots
    theory.html                # LBM theory with KaTeX equations
    implementation.html        # Code architecture + source blocks
    results.html               # Validation plots + field galleries
    css/
      style.css                # CFD Jet theme (dark, cyan/orange accents)
    assets/
      js/
        viewer.js              # Interactive canvas player (stretch goal)
      data/                    # Pre-computed JSON (gitignored)
      images/                  # PNG visualizations (gitignored)
      videos/                  # MP4 animations (gitignored)

  output/                      # Simulation VTK frames (gitignored)
    re100/                     # Per-Re subdirectories
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
Re = u_inflow * NX / nu
tau = 0.5 + 3 * u_inflow * NX / Re

### Boundary conditions
- Zou/He velocity inlet: enforce u = u_inflow at x=0, compute rho from known distributions
- Convective outlet: zero-gradient extrapolation at x=NX-1
- Bounce-back: reverse distribution direction at obstacle nodes
- Periodic: wrap y-coordinate (top connects to bottom)

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

### Grid sizing guide
- NX = 400, NY = 150 recommended (60K fluid nodes)
- Cylinder diameter D = NY/5 = 30 (30 grid points across cylinder -- good resolution)
- Cylinder center at x = NX/4 = 100, y = NY/2 = 75
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

## Reference Commands

```bash
cmake -B build && cmake --build build          # Build solver
./build/LBM_Engine                             # Run default (Re=100)
./build/LBM_Engine 200                         # Run Re=200
./build/LBM_Engine 100 12000                   # Run with custom steps
./build/LBM_Tests                              # Run unit tests

# Batch runner
bash scripts/run_all_re.sh                     # Runs Re = 20, 40, 100, 200

# Post-processing (requires Python + numpy + matplotlib)
python3 scripts/postprocess.py output/re100    # VTK -> PNG frames
python3 scripts/postprocess.py output/re100 --json  # VTK -> JSON for web

# Preview website
python3 -m http.server -d docs 8765
open http://localhost:8765

# Clean
rm -rf build output/re*
```

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
