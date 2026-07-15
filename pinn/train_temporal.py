"""Train the time-parametric PINN on Lid-Driven Cavity (Phase 6.8).

A single network learns the full transient evolution: (x, y, Re_n, t_n) -> (u, v, p).
Input = (x_norm, y_norm, re_norm, t_norm); Fourier features on (x, y), then
concatenate re_norm and t_norm -> 514-dim MLP input.

Usage:
    python3 train_temporal.py                       # multi-Re (100 + 400)
    python3 train_temporal.py --epochs-adam 12000
    python3 train_temporal.py --resume out.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cavity_re100, cavity_re400
from data.loader import grid_coords, flatten_grid
from data.temporal_loader import (
    build_temporal_sensors, build_temporal_collocation, normalize_re, list_frames,
)
from models.pinn import ParametricPINN, predict
from models.losses import total_loss_cavity_temporal

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
N_COLLOC    = 12000     # PDE collocation points (across Re x time)
N_SENSORS   = 600       # sensors per frame (importance sampled)
N_FRAMES    = None      # None = use all available LBM frames
LR_ADAM     = 5e-4
EPOCHS_ADAM = 15000
EPOCHS_LBFGS = 1000
W_PDE       = 5.0
W_DATA      = 5.0
W_BC        = 5.0
W_IC        = 5.0       # initial condition (rest state at t=0)
U_LID       = 0.1
HIDDEN      = 256
N_LAYERS    = 8
N_FREQS     = 128
SIGMA       = 5.0
GRAD_CLIP   = 1.0
RE_VALUES   = [100.0, 400.0]
RE_NORM_VALS = tuple(normalize_re(r) for r in RE_VALUES)


def to_device(t, device):
    return t.to(device, dtype=torch.float32)


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    configs = [cavity_re100(), cavity_re400()]
    base_dir = configs[0].case_dir
    out_dir = os.path.join(os.path.dirname(base_dir), "pinn_temporal")
    os.makedirs(out_dir, exist_ok=True)

    # Frame directories per Re.
    frame_dirs = {cfg.re: os.path.join(cfg.case_dir, "frames") for cfg in configs}

    # Full-grid normalized coords (for initial-condition loss).
    Xn, Yn, _, _ = grid_coords(configs[0])
    coords_all = flatten_grid(Xn, Yn)
    grid_xy_t = to_device(torch.from_numpy(coords_all), device)

    # Sensors: (x, y, re_norm, t_norm) -> (u, v, p) across all frames.
    sens_coords, sens_uvp = build_temporal_sensors(
        configs[0], frame_dirs, RE_VALUES,
        n_per_frame=N_SENSORS, seed=0)
    print(f"Temporal sensors: {sens_coords.shape}  uvp: {sens_uvp.shape}")
    n_frames = [len([f for f in os.listdir(frame_dirs[r])])
                for r in RE_VALUES]
    print(f"Frames per Re (approx): {dict(zip(RE_VALUES, n_frames))}")

    # Build per-Re sensor dict.
    sens_by_re = {}
    offset = 0
    for cfg in configs:
        re_norm = normalize_re(cfg.re)
        # count frames for this Re to slice the concatenated sensor array
        n_f = len(list_frames(frame_dirs[cfg.re]))
        n_this = n_f * N_SENSORS
        sl = slice(offset, offset + n_this)
        coords_re = sens_coords[sl]
        uvp_re = sens_uvp[sl]
        sens_by_re[cfg.re] = {
            "coords": coords_re,
            "u": uvp_re[:, 0],
            "v": uvp_re[:, 1],
            "p": uvp_re[:, 2],
        }
        offset += n_this
        print(f"  Re={cfg.re}: {coords_re.shape} sensors")

    # Collocation per Re.
    colloc_by_re = {}
    for cfg in configs:
        col = build_temporal_collocation(
            configs[0], RE_VALUES, N_COLLOC, seed=1, t_min=0.0, t_max=1.0)
        colloc_by_re[cfg.re] = col
    print(f"Collocation per Re: {N_COLLOC}")

    # Model: 4 inputs via ParametricPINN(n_params=2).
    model = ParametricPINN(n_params=2, hidden=HIDDEN, n_layers=N_LAYERS,
                           n_freqs=N_FREQS, sigma=SIGMA).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Resumed from {args.resume}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, factor=0.5, patience=500, min_lr=1e-5)

    t0 = time.time()
    history = {"epoch": [], "loss": [], "pde": [], "data": [], "bc": [], "ic": []}

    print(f"\n--- Adam training ({EPOCHS_ADAM} epochs) ---")
    for ep in range(1, EPOCHS_ADAM + 1):
        optimizer_adam.zero_grad()
        L, Lpde, Ldata, Lbc, Lic = total_loss_cavity_temporal(
            model, colloc_by_re, sens_by_re, grid_xy_t, u_lid=U_LID,
            device=device, w_pde=W_PDE, w_data=W_DATA, w_bc=W_BC, w_ic=W_IC,
            re_norm_vals=RE_NORM_VALS)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer_adam.step()
        scheduler.step(L.item())

        if ep % 200 == 0 or ep == 1:
            lr = optimizer_adam.param_groups[0]["lr"]
            print(f"  epoch {ep:5d}  loss={L.item():.6f}  "
                  f"pde={Lpde.item():.6f}  data={Ldata.item():.6f}  "
                  f"bc={Lbc.item():.6f}  ic={Lic.item():.6f}  lr={lr:.2e}")
            history["epoch"].append(ep)
            history["loss"].append(L.item())
            history["pde"].append(Lpde.item())
            history["data"].append(Ldata.item())
            history["bc"].append(Lbc.item())
            history["ic"].append(Lic.item())

    adam_time = time.time() - t0
    print(f"Adam phase done in {adam_time:.1f}s")

    # L-BFGS fine-tune.
    print(f"\n--- L-BFGS fine-tune ({EPOCHS_LBFGS} steps) ---")
    t1 = time.time()
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=EPOCHS_LBFGS, history_size=100,
        line_search_fn="strong_wolfe")

    def closure():
        optimizer_lbfgs.zero_grad()
        L, _, _, _, _ = total_loss_cavity_temporal(
            model, colloc_by_re, sens_by_re, grid_xy_t, u_lid=U_LID,
            device=device, w_pde=W_PDE, w_data=W_DATA, w_bc=W_BC, w_ic=W_IC,
            re_norm_vals=RE_NORM_VALS)
        L.backward()
        return L

    L_lbfgs = optimizer_lbfgs.step(closure)
    lbfs_time = time.time() - t1
    print(f"L-BFGS done in {lbfs_time:.1f}s  final loss={L_lbfgs.item():.6f}")

    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Save.
    model_path = os.path.join(out_dir, "model_temporal.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "n_params": 2,
        "n_freqs": model.n_freqs,
        "sigma": model.sigma,
        "hidden": HIDDEN,
        "n_layers": N_LAYERS,
    }, model_path)
    print(f"Model saved: {model_path}")

    np.savez(os.path.join(out_dir, "loss_history_temporal.npz"), **history)
    print(f"Loss history saved: {out_dir}/loss_history_temporal.npz")

    print(f"\n=== SUMMARY (Temporal) ===")
    print(f"Device:  {device}")
    print(f"Params:  {n_params}")
    print(f"Time:    {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-adam", type=int, default=EPOCHS_ADAM)
    parser.add_argument("--epochs-lbfgs", type=int, default=EPOCHS_LBFGS)
    parser.add_argument("--n-colloc", type=int, default=N_COLLOC)
    parser.add_argument("--n-sensors", type=int, default=N_SENSORS)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--lr", type=float, default=LR_ADAM)
    parser.add_argument("--w-ic", type=float, default=W_IC)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    EPOCHS_ADAM = args.epochs_adam
    EPOCHS_LBFGS = args.epochs_lbfgs
    N_COLLOC = args.n_colloc
    N_SENSORS = args.n_sensors
    HIDDEN = args.hidden
    LR_ADAM = args.lr
    W_IC = args.w_ic

    train(args)
