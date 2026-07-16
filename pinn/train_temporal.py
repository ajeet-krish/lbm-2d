"""Train the time-parametric PINN on Lid-Driven Cavity (Phase 6.8 + improvements).

A single network learns the full transient evolution: (x, y, Re_n, t_n) -> (u, v, p).
Input = (x_norm, y_norm, re_norm, t_norm); Fourier features on (x, y), then
concatenate re_norm and t_norm -> 514-dim MLP input.

Improvements over Phase 6.8 baseline:
  - Pressure-Poisson residual (fixes near-constant pressure prediction)
  - Vorticity-transport residual (sharpens vortex-core accuracy)
  - Curriculum learning on early frames (eases initial transient difficulty)
  - Adaptive importance sampling (residual-based sensor resampling)
  - Extended Re range to 1000 (data already simulated)

Usage:
    python3 train_temporal.py                       # multi-Re (100 + 400 + 1000)
    python3 train_temporal.py --epochs-adam 15000
    python3 train_temporal.py --resume out.pt
    python3 train_temporal.py --no-curriculum      # disable curriculum learning
    python3 train_temporal.py --no-adaptive        # disable adaptive sampling
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cavity_re100, cavity_re400, cavity_re1000
from data.loader import grid_coords, flatten_grid
from data.temporal_loader import (
    build_temporal_sensors, build_temporal_collocation, normalize_re,
    list_frames, adaptive_resample_sensors,
)
from models.pinn import ParametricPINN, predict
from models.losses import total_loss_cavity_temporal, _split

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
N_COLLOC    = 2000      # PDE collocation points per Re (batched at 500)
N_SENSORS   = 600       # sensors per frame (importance sampled)
N_FRAMES    = None      # None = use all available LBM frames
LR_ADAM     = 5e-4
EPOCHS_ADAM = 15000
EPOCHS_LBFGS = 1000
W_PDE       = 5.0
W_DATA      = 5.0
W_BC        = 5.0
W_IC        = 5.0       # initial condition (rest state at t=0)
W_PP        = 1.0       # pressure-Poisson residual (scale-normalized)
W_VORT      = 0.5       # vorticity-transport residual (scale-normalized)
U_LID       = 0.1
HIDDEN      = 256
N_LAYERS    = 8
N_FREQS     = 128
SIGMA       = 5.0
SIGMAS      = None      # None = single-scale; set tuple for multi-scale e.g. (1,5,20)
GRAD_CLIP   = 1.0
RE_VALUES   = [100.0, 400.0, 1000.0]
RE_NORM_VALS = tuple(normalize_re(r) for r in RE_VALUES)
USE_CURRICULUM = True   # progressive time-window expansion
USE_ADAPTIVE  = True    # residual-based sensor resampling
ADAPTIVE_INTERVAL = 2000  # re-sample every N epochs
PRETRAIN_EPOCHS = 1000   # data-only pretraining to avoid constant-collapse
P_SCALE        = 10.0     # pressure loss weight (scale-normalized by p std)
CURRICULUM_STAGES = [(0.4, 5000), (0.7, 10000), (1.0, 15000)]  # (t_max, epoch)


def to_device(t, device):
    return t.to(device, dtype=torch.float32)


def data_loss_scaled(model, sens_by_re, device, w_p_scale=10.0):
    """Data loss with scale-aware pressure weighting.

    Pressure has near-zero mean and small variance in lattice units, so a raw
    MSE on p is minimized by predicting the mean (collapse). We divide the
    pressure residual by the target std so the network must learn p variation.
    """
    L = 0.0
    n = 0
    for re, sens in sens_by_re.items():
        coords_t = torch.from_numpy(sens["coords"]).float().to(device)
        u_t = torch.from_numpy(sens["u"]).float().to(device)
        v_t = torch.from_numpy(sens["v"]).float().to(device)
        p_t = torch.from_numpy(sens["p"]).float().to(device)
        u_pred, v_pred, p_pred = _split(model(coords_t))
        p_std = p_t.std().clamp(min=1e-4)
        L = (L + F.mse_loss(u_pred, u_t) + F.mse_loss(v_pred, v_t)
             + w_p_scale * F.mse_loss(p_pred, p_t) / (p_std ** 2))
        n += 1
    return L / n


def get_curriculum_t_max(epoch: int) -> float:
    """Return the max normalized time for the current curriculum stage."""
    t_max = 0.0
    for stage_t_max, stage_epoch in CURRICULUM_STAGES:
        if epoch <= stage_epoch:
            t_max = stage_t_max
            break
    else:
        t_max = 1.0
    return t_max


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    configs = [cavity_re100(), cavity_re400(), cavity_re1000()]
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
                           n_freqs=N_FREQS, sigma=SIGMA, sigmas=SIGMAS).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Resumed from {args.resume}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")

    optimizer_adam = torch.optim.Adam(
        model.parameters(), lr=LR_ADAM, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, factor=0.5, patience=500, min_lr=1e-5)

    t0 = time.time()
    history = {"epoch": [], "loss": [], "pde": [], "pp": [], "vort": [],
               "data": [], "bc": [], "ic": []}

    # Phase 0: Data-only pretraining to avoid constant-collapse.
    # The cavity mean flow is ~zero, so a trivial constant solution minimizes
    # both data (mean u ~ 0) and PDE (u=const satisfies NS). Pretraining on
    # data alone forces the network to learn spatial structure first.
    if PRETRAIN_EPOCHS > 0:
        print(f"\n--- Data-only pretraining ({PRETRAIN_EPOCHS} epochs) ---")
        for ep in range(1, PRETRAIN_EPOCHS + 1):
            optimizer_adam.zero_grad()
            Ldata = data_loss_scaled(model, sens_by_re, device, w_p_scale=P_SCALE)
            Ldata.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer_adam.step()
            if ep % 200 == 0 or ep == 1:
                print(f"  pretrain epoch {ep:5d}  data={Ldata.item():.6f}")
        print(f"Pretraining done. Data loss: {Ldata.item():.6f}")

    print(f"\n--- Adam training ({EPOCHS_ADAM} epochs) ---")
    print(f"  Curriculum: {USE_CURRICULUM}  Adaptive sampling: {USE_ADAPTIVE}")
    for ep in range(1, EPOCHS_ADAM + 1):
        # Curriculum: filter sensors to current time window.
        if USE_CURRICULUM:
            t_max = get_curriculum_t_max(ep)
            sens_by_re_cur = {}
            for re, sens in sens_by_re.items():
                mask = sens["coords"][:, 3] <= t_max + 1e-6
                sens_by_re_cur[re] = {
                    "coords": sens["coords"][mask],
                    "u": sens["u"][mask],
                    "v": sens["v"][mask],
                    "p": sens["p"][mask],
                }
        else:
            sens_by_re_cur = sens_by_re

        optimizer_adam.zero_grad()
        L, Lpde, Lpp, Lvort, Ldata, Lbc, Lic = total_loss_cavity_temporal(
            model, colloc_by_re, sens_by_re_cur, grid_xy_t, u_lid=U_LID,
            device=device, w_pde=W_PDE, w_data=W_DATA, w_bc=W_BC, w_ic=W_IC,
            w_pp=W_PP, w_vort=W_VORT, re_norm_vals=RE_NORM_VALS)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer_adam.step()
        scheduler.step(L.item())

        # Adaptive sampling: re-sample sensors based on current residual.
        if USE_ADAPTIVE and ep % ADAPTIVE_INTERVAL == 0:
            print(f"  [epoch {ep}] Adaptive resampling sensors...")
            sens_by_re = adaptive_resample_sensors(
                model, configs, frame_dirs, RE_VALUES, sens_by_re,
                n_per_frame=N_SENSORS, device=device)

        if ep % 200 == 0 or ep == 1:
            lr = optimizer_adam.param_groups[0]["lr"]
            print(f"  epoch {ep:5d}  loss={L.item():.6f}  "
                  f"pde={Lpde.item():.6f}  pp={Lpp.item():.6f}  "
                  f"vort={Lvort.item():.6f}  data={Ldata.item():.6f}  "
                  f"bc={Lbc.item():.6f}  ic={Lic.item():.6f}  lr={lr:.2e}")
            history["epoch"].append(ep)
            history["loss"].append(L.item())
            history["pde"].append(Lpde.item())
            history["pp"].append(Lpp.item())
            history["vort"].append(Lvort.item())
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
        L, _, _, _, _, _, _ = total_loss_cavity_temporal(
            model, colloc_by_re, sens_by_re, grid_xy_t, u_lid=U_LID,
            device=device, w_pde=W_PDE, w_data=W_DATA, w_bc=W_BC, w_ic=W_IC,
            w_pp=W_PP, w_vort=W_VORT, re_norm_vals=RE_NORM_VALS)
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
        "sigma": model.sigma if model.sigmas is None else list(model.sigmas),
        "sigmas": list(model.sigmas) if model.sigmas is not None else None,
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
    parser.add_argument("--w-data", type=float, default=W_DATA)
    parser.add_argument("--w-pde", type=float, default=W_PDE)
    parser.add_argument("--w-pp", type=float, default=W_PP)
    parser.add_argument("--w-vort", type=float, default=W_VORT)
    parser.add_argument("--pretrain", type=int, default=PRETRAIN_EPOCHS,
                        help="Data-only pretraining epochs")
    parser.add_argument("--sigmas", type=str, default=None,
                        help="Multi-scale Fourier sigmas as comma list, e.g. '1,5,20'")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning on early frames")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive importance sampling")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    EPOCHS_ADAM = args.epochs_adam
    EPOCHS_LBFGS = args.epochs_lbfgs
    N_COLLOC = args.n_colloc
    N_SENSORS = args.n_sensors
    HIDDEN = args.hidden
    LR_ADAM = args.lr
    W_IC = args.w_ic
    W_DATA = args.w_data
    W_PDE = args.w_pde
    W_PP = args.w_pp
    W_VORT = args.w_vort
    PRETRAIN_EPOCHS = args.pretrain
    USE_CURRICULUM = not args.no_curriculum
    USE_ADAPTIVE = not args.no_adaptive
    if args.sigmas:
        SIGMAS = tuple(float(s) for s in args.sigmas.split(","))

    train(args)
