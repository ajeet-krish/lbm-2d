# Graph Report - .  (2026-07-14)

## Corpus Check
- Large corpus: 161 files · ~1,304,069 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 624 nodes · 870 edges · 94 communities (34 shown, 60 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 85 edges (avg confidence: 0.82)
- Token cost: 178,357 input · 15,174 output

## Community Hubs (Navigation)
- [[_COMMUNITY_MRT Collision & Cavity|MRT Collision & Cavity]]
- [[_COMMUNITY_AMR & Bounce-Back Geometry|AMR & Bounce-Back Geometry]]
- [[_COMMUNITY_Urban Canyon Simulation|Urban Canyon Simulation]]
- [[_COMMUNITY_PINN Data Loader|PINN Data Loader]]
- [[_COMMUNITY_LBM Core Types & Tests|LBM Core Types & Tests]]
- [[_COMMUNITY_WASMJS Interface|WASM/JS Interface]]
- [[_COMMUNITY_WASM Loader (Emscripten)|WASM Loader (Emscripten)]]
- [[_COMMUNITY_PINN Loss Functions|PINN Loss Functions]]
- [[_COMMUNITY_Unit Test Suite|Unit Test Suite]]
- [[_COMMUNITY_PINN Configuration|PINN Configuration]]
- [[_COMMUNITY_Postprocessing Pipeline|Postprocessing Pipeline]]
- [[_COMMUNITY_Orifice Plate Simulation|Orifice Plate Simulation]]
- [[_COMMUNITY_LBM Solver Core|LBM Solver Core]]
- [[_COMMUNITY_Periodic Hills Simulation|Periodic Hills Simulation]]
- [[_COMMUNITY_PINN Pipeline Docs|PINN Pipeline Docs]]
- [[_COMMUNITY_Downwash Simulation|Downwash Simulation]]
- [[_COMMUNITY_Rotating Cylinder|Rotating Cylinder]]
- [[_COMMUNITY_Side-by-Side Cylinders|Side-by-Side Cylinders]]
- [[_COMMUNITY_Cylinder Near Wall|Cylinder Near Wall]]
- [[_COMMUNITY_Flat Plate Simulation|Flat Plate Simulation]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]

## God Nodes (most connected - your core abstractions)
1. `UrbanParams` - 37 edges
2. `AMRBlock` - 30 edges
3. `AMRGrid` - 24 edges
4. `TEST()` - 23 edges
5. `execute_time_step()` - 21 edges
6. `BounceBackGeometry` - 18 edges
7. `OrificeParams` - 15 edges
8. `EMSCRIPTEN_KEEPALIVE` - 14 edges
9. `save_forces_jsonl()` - 13 edges
10. `save_meta_json()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Centerline Velocity Profiles` --references--> `Navier-Stokes Equations`  [INFERRED]
  docs/assets/images/cavity/profiles.png → docs/theory.html
- `PINN vs LBM Comparison` --references--> `Navier-Stokes Equations`  [INFERRED]
  docs/assets/images/cylinder/pinn_steady_comparison.png → docs/theory.html
- `CaseConfig` --uses--> `CaseConfig`  [INFERRED]
  pinn/data/loader.py → pinn/config.py
- `ndarray` --uses--> `CaseConfig`  [INFERRED]
  pinn/data/loader.py → pinn/config.py
- `CaseConfig` --uses--> `CaseConfig`  [INFERRED]
  pinn/evaluate.py → pinn/config.py

## Import Cycles
- None detected.

## Communities (94 total, 60 thin omitted)

### Community 0 - "MRT Collision & Cavity"
Cohesion: 0.09
Nodes (42): MRTParams, CavityParams, num_steps, tau, u_lid, vtk_interval, compute_params(), main() (+34 more)

### Community 1 - "AMR & Bounce-Back Geometry"
Cohesion: 0.08
Nodes (26): BounceBackGeometry, AMRBlock, bb_geom, dx, f, f_next, fx_body, fy_body (+18 more)

### Community 2 - "Urban Canyon Simulation"
Cohesion: 0.06
Nodes (40): compute_side_params(), compute_topdown_params(), LBMCapabilities, layout_side(), main(), place_side_obstacles(), place_topdown_obstacles(), UrbanParams (+32 more)

### Community 3 - "PINN Data Loader"
Cohesion: 0.09
Nodes (32): flatten_grid(), grid_coords(), load_case_frame(), load_forces(), load_frame(), make_boundary(), make_collocation(), Data loading and sampling utilities for the PINN surrogate suite.  torch-free (n (+24 more)

### Community 4 - "LBM Core Types & Tests"
Cohesion: 0.06
Nodes (24): BounceBackGeometry, cx, cy, has_moving_wall, is_polygon, omega, poly_vertices, radius (+16 more)

### Community 5 - "WASM/JS Interface"
Cohesion: 0.17
Nodes (26): compute_equilibrium(), compute_macros(), enforce_inflow(), enforce_outflow(), naca_coords(), node_index(), place_polygon(), place_shape() (+18 more)

### Community 6 - "WASM Loader (Emscripten)"
Cohesion: 0.13
Nodes (22): abort(), convertReturnValue(), createWasm(), EmscriptenEH, EmscriptenSjLj, ExitStatus, findWasmBinary(), getBinarySync() (+14 more)

### Community 7 - "PINN Loss Functions"
Cohesion: 0.19
Nodes (22): device, bc_loss_cylinder(), bc_loss_inflow(), bc_loss_outlet(), bc_loss_walls(), data_loss(), _grad(), _on_device() (+14 more)

### Community 8 - "Unit Test Suite"
Cohesion: 0.10
Nodes (21): AllDirectionsCovered, BounceBackTest, CylinderPlacement, EquilibriumTest, ForceTest, IndexTest, MacrosTest, MassConservation (+13 more)

### Community 9 - "PINN Configuration"
Cohesion: 0.12
Nodes (16): CaseConfig, cylinder_re100(), downsample_factor(), from_meta(), _geometry_for(), ndarray, Configuration and case metadata for the PINN surrogate suite.  This module is to, Convenience constructor for the first training case. (+8 more)

### Community 10 - "Postprocessing Pipeline"
Cohesion: 0.25
Nodes (17): compute_strouhal(), _detect_shape(), _list_frames(), _load_frame(), _load_meta(), main(), make_video(), _overlay_obstacles() (+9 more)

### Community 11 - "Orifice Plate Simulation"
Cohesion: 0.16
Nodes (17): compute_params(), LBMCapabilities, string, vector, main(), OrificeParams, config, hole_width (+9 more)

### Community 12 - "LBM Solver Core"
Cohesion: 0.14
Nodes (15): LBM_Engine, PINN Surrogate Suite, compute_params(), vector, ForceHistory, cd, cl, t (+7 more)

### Community 13 - "Periodic Hills Simulation"
Cohesion: 0.17
Nodes (15): compute_body_force(), compute_params(), LBMCapabilities, HillParams, body_force_x, H, h_max, L (+7 more)

### Community 14 - "PINN Pipeline Docs"
Cohesion: 0.17
Nodes (14): Lattice Boltzmann Method, Physics-Informed Neural Network, Data Loader, Evaluation Script, ONNX Export Script, PINN Losses, PINN Model (MLP), Results Plotter (+6 more)

### Community 15 - "Downwash Simulation"
Cohesion: 0.18
Nodes (11): DownwashParams, gap, h_low, h_tall, low_x0, num_steps, save_interval, tall_x0 (+3 more)

### Community 16 - "Rotating Cylinder"
Cohesion: 0.18
Nodes (11): RotatingParams, cx_cyl, cy_cyl, length_scale, num_steps, omega, radius, save_interval (+3 more)

### Community 17 - "Side-by-Side Cylinders"
Cohesion: 0.18
Nodes (11): SideBySideParams, cx, cy1, cy2, length_scale, num_steps, radius, save_interval (+3 more)

### Community 18 - "Cylinder Near Wall"
Cohesion: 0.20
Nodes (10): GapParams, cx_cyl, cy_cyl, gap, length_scale, num_steps, radius, save_interval (+2 more)

### Community 19 - "Flat Plate Simulation"
Cohesion: 0.20
Nodes (10): FlatPlateParams, aoa_deg, chord, cx, cy, length_scale, num_steps, save_interval (+2 more)

### Community 20 - "Community 20"
Cohesion: 0.22
Nodes (9): SquareCylParams, cx, cy, length_scale, num_steps, save_interval, side, tau (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.43
Nodes (7): load_lbm_data(), main(), plot_cd_vs_aoa(), plot_cl_vs_aoa(), plot_drag_polar(), Load force coefficients from saved simulations., thin_airfoil_theory()

### Community 22 - "Community 22"
Cohesion: 0.25
Nodes (8): StepParams, h_inlet, h_step, length_scale, num_steps, save_interval, tau, u_max

### Community 23 - "Community 23"
Cohesion: 0.29
Nodes (7): Centerline Velocity Profiles, PINN vs LBM Comparison, BGK Collision Operator, Boltzmann Equation, Chapman-Enskog Expansion, D2Q9 Lattice, Navier-Stokes Equations

### Community 24 - "Community 24"
Cohesion: 0.33
Nodes (5): plot_comparison(), plot_loss_history(), Generate 3-panel comparison plot: LBM / PINN / Error delta.  Also generates loss, 3-panel: LBM / PINN / absolute error delta., Loss convergence curves.

### Community 25 - "Community 25"
Cohesion: 0.47
Nodes (5): compute_strouhal(), load_forces(), main(), Load forces.jsonl, return (steps, cd/cl or fx/fy) arrays., Compute Strouhal number from fy time series using FFT.          Uses zero-crossi

### Community 26 - "Community 26"
Cohesion: 0.67
Nodes (5): pair, vector, naca_coords(), point_in_polygon(), transform_points()

### Community 27 - "Community 27"
Cohesion: 0.67
Nodes (3): Streamline plot of flow through an orifice plate (2p configuration), Velocity magnitude contour plot of flow through an orifice plate (3p configuration), Streamline plot of flow through an orifice plate (3p configuration)

### Community 29 - "Community 29"
Cohesion: 0.67
Nodes (3): Bounce-Back Boundary Condition, Oke 1988 Flow Regimes, Side-View Canyon Analysis

## Knowledge Gaps
- **289 isolated node(s):** `device`, `batch_postprocess.sh script`, `run_ahmed.sh script`, `run_all_cases.sh script`, `run_downwash.sh script` (+284 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **60 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TEST()` connect `Unit Test Suite` to `MRT Collision & Cavity`, `LBM Core Types & Tests`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `TEST()` (e.g. with `execute_time_step()` and `place_cylinder()`) actually correct?**
  _`TEST()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `execute_time_step()` (e.g. with `main()` and `main()`) actually correct?**
  _`execute_time_step()` has 13 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Configuration and case metadata for the PINN surrogate suite.  This module is to`, `Resolved parameters for one simulation case.`, `Kinematic viscosity in lattice units: nu = (tau - 0.5) / 3.` to the rest of the system?**
  _331 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MRT Collision & Cavity` be split into smaller, more focused modules?**
  _Cohesion score 0.08784313725490196 - nodes in this community are weakly interconnected._
- **Should `AMR & Bounce-Back Geometry` be split into smaller, more focused modules?**
  _Cohesion score 0.07673469387755102 - nodes in this community are weakly interconnected._
- **Should `Urban Canyon Simulation` be split into smaller, more focused modules?**
  _Cohesion score 0.06219512195121951 - nodes in this community are weakly interconnected._