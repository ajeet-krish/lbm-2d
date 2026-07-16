# PINN Surrogate Suite -- Physics-Informed Neural Networks

A parametric surrogate suite in `pinn/`. Trains PyTorch PINNs on C++ LBM
output as hybrid data-physics surrogates, validated against the solver, and
deployed as real-time ONNX Runtime Web inference demos. Mirrors the SciML
R&D pipeline at NASA, Rolls-Royce, and F1 teams.

**Zero changes to the existing C++ solver.** All code lives under `pinn/`.

## Hardware

Apple Silicon Mac (tested on M5). Training uses `torch.device("mps")`
(Metal Performance Shaders). No NVIDIA GPU required.

## What is a Parametric PINN?

A standard PINN learns `(x, y) -> (u, v, p)` for a single flow condition.
A **parametric PINN** adds physical parameters as inputs:

```
[x, y, Re, hole_w, ...] --> [Neural Network] --> [u, v, p]
```

This creates a **continuous design-space surrogate**: drag a slider to change
Reynolds number or orifice diameter, and the network predicts the updated flow
field instantly -- no retraining needed.

## Implementation Roadmap

| Phase | Case | Input | Data Status | Portfolio Demo |
|-------|------|-------|-------------|----------------|
| 6.0 | Setup | -- | Done | -- |
| 6.1 | Cylinder Re=100 | (x, y) | Done | 3-panel comparison |
| 6.2 | Website integration | -- | Done | Error delta map |
| **6.3** | **Cavity** | **(x, y, Re)** | **Re=100, 400 on disk** | **Re slider -> vortex shift** |
| **6.4** | **Backward step** | **(x, y, Re)** | **Re-run Re=100 needed** | **Re slider -> Xr/H** |
| **6.5** | **Orifice plate** | **(x, y, hole_w, n_plates)** | **New simulations needed** | **Diameter slider -> K** |
| 6.6 | ONNX export + WASM | -- | -- | Live browser inference |
| 6.7 | Ablation study | -- | -- | Methodology write-up |

### Phase 6.3: Cavity (FIRST -- all data exists)

- **Data:** Re=100 (51 frames, 128x128, steady) + Re=400 (51 frames, steady)
- **Architecture:** `(x, y, Re_n) -> (u, v, p)` with Re normalized to [0, 1]
- **BCs:** All no-slip walls + moving lid (u_lid = 0.1). No inflow/outlet.
- **Training:** Multi-Re -- sample collocation at both Re=100 and Re=400
- **Demo:** "Drag Re slider from 100 to 400, watch vortex center migrate"

### Phase 6.4: Backward-Facing Step (SECOND)

- **Data:** Re-run Re=100 with pressure output (~2hr). Re=400 fallback.
- **Architecture:** Same 3-input as cavity. Add parabolic inlet BC loss.
- **Demo:** "Drag Re slider, watch reattachment length grow"

### Phase 6.5: Orifice Plate (THIRD -- highest portfolio impact)

- **Data:** New Re sweep (50, 100, 200, 500) + hole-width sweep (10-80) for 1p1h
- **Architecture:** `(x, y, Re_n, hole_w_n) -> (u, v, p)` with rectangular plate encoding
- **Demo:** "Drag orifice diameter slider, see loss coefficient K change"

## Architecture

### Steady-State PINN (single case)
```
Input: (x, y) normalized to [-1, 1]
  FC 2 -> 64 (x8 hidden, tanh) -> 3
Output: (u, v, p)
```

### Parametric PINN (multi-case)
```
Input: (x, y, param1, param2, ...) normalized
  FC (2+n_params) -> 128 (x8 hidden, tanh) -> 3
Output: (u, v, p)
```

### Hybrid Loss
```
L = w_data * MSE(pred, LBM) + w_pde * NS_residual + w_bc * BC_loss
```

PDE residual = steady incompressible Navier-Stokes. All derivatives via
`torch.autograd`. BC losses vary by case (cavity: walls + lid; cylinder:
inflow + outlet + walls + surface).

## Quick start

```bash
# Install dependencies (Python >= 3.10 recommended)
pip3 install -r requirements.txt

# Smoke-test data loader (numpy only, no torch needed)
python3 data/loader.py

# Train cylinder PINN (single case, original)
python3 cases/cylinder/train.py

# Train cavity parametric PINN (multi-Re)
python3 cases/cavity/train_steady.py

# Train cavity at single Re
python3 cases/cavity/train_steady.py --single-re 100
python3 cases/cavity/train_steady.py --single-re 400

# Train cavity time-parametric PINN (spatio-temporal surrogate)
python3 cases/cavity/train_temporal.py
```

## Directory layout

```
pinn/
  README.md              # This file
  requirements.txt       # torch, numpy, matplotlib, scipy, onnx, onnxruntime
  config.py              # CaseConfig: meta -> grid dims, ds, geometry (shared)
  data/                  # Shared data loaders
    loader.py            # Load frame*.json -> numpy; steady collocation/sensors
    temporal_loader.py   # Temporal (4-D) sensor + collocation loaders
  models/                # Shared network architectures + losses
    pinn.py              # PINN + ParametricPINN MLP; Fourier feature layers
    losses.py            # PDE residual, BC loss (cavity/cylinder), data loss
  export/                # Multi-case export (shared)
    export_web_data.py   # LBM frames -> binary for web viewer (all cases)
  cases/                 # Per-case organization
    cavity/              # Lid-driven cavity surrogate
      train_steady.py    # Phase 6.3: parametric steady-state PINN
      train_temporal.py  # Phase 6.8: time-parametric (spatio-temporal) PINN
      export_sweep.py    # Cavity steady-state sweep + ONNX export
      export_temporal.py # Cavity temporal ONNX + binary frame export
      plot_results.py    # 3-panel (LBM / PINN / Error delta)
      plot_loss_convergence.py  # Training loss convergence plot
      plot_temporal_l2.py       # Frame-by-frame temporal L2 profile
      logs/               # Training logs (gitignored)
    cylinder/            # Cylinder wake surrogate
      train.py           # Phase 6.1: steady-state PINN (single Re=100)
      evaluate.py        # Inference on full grid -> numpy fields

# Trained model artifacts live in output/ (gitignored), organized by case:
output/
  cavity/
    pinn/
      stable/            # Steady-state parametric PINN (Phase 6.3)
        v1/              # 64-wide, no Fourier (baseline)
        v2/              # 256-wide, no Fourier
        v3/              # Fourier multi-scale (current best, ACTIVE)
          model.pt
          prediction_re{100,400}.npz
          loss_history.npz
      temporal/          # Time-parametric PINN (Phase 6.8)
        v1/              # Current temporal model (4-input: x, y, Re, t)
          model.pt
          loss_history.npz
  cylinder/
    re100/
      pinn/              # Phase 6.1 artifacts (model.pt, prediction.npz)
```

## Key references

- Raissi, Perdikaris, Karniadakis (2019), "Physics-informed neural networks", JCP.
- Cavity validation: Ghia, Ghia & Shin (1982).
- Step validation: Armaly et al. (1983).
- Orifice validation: ISO 5167, Idelchik (2006).
