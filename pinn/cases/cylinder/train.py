"""Train a steady-state hybrid PINN on Cylinder Re=100.

Usage:
    python3 train.py                        # default: cylinder Re=100, steady
    python3 train.py --case cylinder_re100  # same
    python3 train.py --epochs-adam 8000     # longer Adam phase
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import CaseConfig, cylinder_re100
from data.loader import (
    load_frame, grid_coords, flatten_grid,
    make_collocation, make_boundary, subsample_sensors,
)
from models.pinn import PINN, predict
from models.losses import total_loss, pde_loss, data_loss
from models.losses import bc_loss_inflow, bc_loss_outlet, bc_loss_walls, bc_loss_cylinder

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
N_COLLOC    = 5000      # interior collocation points for PDE residual
N_SENSORS   = 1000      # sparse sensor subsample for data loss
N_BC        = 400       # cylinder boundary points
N_WALL_BC   = 200       # wall BC points (top + bottom + inlet + outlet)
LR_ADAM     = 1e-3
EPOCHS_ADAM = 5000
EPOCHS_LBFGS = 500
W_PDE       = 1.0
W_DATA      = 10.0
W_BC        = 5.0
FRAME_NAME  = "frame_18000.json"


def to_device(t, device):
    return t.to(device, dtype=torch.float32)


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = cylinder_re100()
    case_dir = os.path.join(cfg.case_dir, "pinn")
    os.makedirs(case_dir, exist_ok=True)

    # --- Load data -------------------------------------------------------
    fr = load_frame(os.path.join(cfg.case_dir, "frames", FRAME_NAME))
    Xn, Yn, _, _ = grid_coords(cfg)
    coords_all = flatten_grid(Xn, Yn)                          # (3800, 2)

    obstacle = fr["obstacle"]
    u_lb = fr["u"].ravel()
    v_lb = fr["v"].ravel()
    p_lb = fr["p"].ravel()

    colloc = make_collocation(cfg, N_COLLOC, seed=0, obstacle=obstacle)
    bnd    = make_boundary(cfg, N_BC, seed=1)
    sens   = subsample_sensors(
        fr["u"], fr["v"], fr["p"], obstacle, N_SENSORS, seed=2,
        normalized_coords=coords_all,
    )

    # BC points are generated internally by bc_loss_* functions.

    # Geometry in normalized coords
    cx_norm = cfg.geometry["cx"] / (cfg.nx - 1) * 2 - 1
    cy_norm = cfg.geometry["cy"] / (cfg.ny - 1) * 2 - 1
    # Normalized semi-axes: radius in physical coords, scaled by domain
    r_norm_x = cfg.geometry["radius"] * cfg.ds / cfg.NX * 2
    r_norm_y = cfg.geometry["radius"] * cfg.ds / cfg.NY * 2

    print(f"Geometry: cx_norm={cx_norm:.4f}  cy_norm={cy_norm:.4f}  "
          f"r_norm_x={r_norm_x:.4f}  r_norm_y={r_norm_y:.4f}")
    print(f"Collocation: {colloc.shape}  Sensors: {sens['coords'].shape}  "
          f"Boundary: {bnd.shape}")

    # --- To tensors on device --------------------------------------------
    colloc_t = to_device(torch.from_numpy(colloc), device)
    sens_coords_t = to_device(torch.from_numpy(sens["coords"]), device)
    sens_u_t = to_device(torch.from_numpy(sens["u"]), device)
    sens_v_t = to_device(torch.from_numpy(sens["v"]), device)

    # --- Model + optimizer -----------------------------------------------
    model = PINN(hidden=64, n_layers=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, factor=0.5, patience=500, min_lr=1e-5
    )

    # --- Training loop ---------------------------------------------------
    t0 = time.time()
    history = {"epoch": [], "loss": [], "pde": [], "data": [], "bc": []}

    print(f"\n--- Adam training ({EPOCHS_ADAM} epochs) ---")
    for ep in range(1, EPOCHS_ADAM + 1):
        optimizer_adam.zero_grad()
        L, Lpde, Ldata, Lbc = total_loss(
            model, colloc_t, sens_coords_t, sens_u_t, sens_v_t,
            re=cfg.re, u_inflow=cfg.u_inflow,
            cx_norm=cx_norm, cy_norm=cy_norm,
            r_norm_x=r_norm_x, r_norm_y=r_norm_y,
            device=device, w_pde=W_PDE, w_data=W_DATA, w_bc=W_BC,
        )
        L.backward()
        optimizer_adam.step()
        scheduler.step(L.item())

        if ep % 200 == 0 or ep == 1:
            lr = optimizer_adam.param_groups[0]["lr"]
            print(f"  epoch {ep:5d}  loss={L.item():.6f}  "
                  f"pde={Lpde.item():.6f}  data={Ldata.item():.6f}  "
                  f"bc={Lbc.item():.6f}  lr={lr:.2e}")
            history["epoch"].append(ep)
            history["loss"].append(L.item())
            history["pde"].append(Lpde.item())
            history["data"].append(Ldata.item())
            history["bc"].append(Lbc.item())

    adam_time = time.time() - t0
    print(f"Adam phase done in {adam_time:.1f}s")

    # --- L-BFGS fine-tune ------------------------------------------------
    print(f"\n--- L-BFGS fine-tune ({EPOCHS_LBFGS} steps) ---")
    t1 = time.time()
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0,
        max_iter=EPOCHS_LBFGS, history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        L, Lpde, Ldata, Lbc = total_loss(
            model, colloc_t, sens_coords_t, sens_u_t, sens_v_t,
            re=cfg.re, u_inflow=cfg.u_inflow,
            cx_norm=cx_norm, cy_norm=cy_norm,
            r_norm_x=r_norm_x, r_norm_y=r_norm_y,
            device=device, w_pde=W_PDE, w_data=W_DATA, w_bc=W_BC,
        )
        L.backward()
        return L

    L_lbfgs = optimizer_lbfgs.step(closure)
    lbfgs_time = time.time() - t1
    print(f"L-BFGS done in {lbfgs_time:.1f}s  final loss={L_lbfgs.item():.6f}")

    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # --- Save model ------------------------------------------------------
    model_path = os.path.join(case_dir, "model_steady.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # --- Inference on full grid -------------------------------------------
    model.eval()
    coords_t = to_device(torch.from_numpy(coords_all), device)
    with torch.no_grad():
        u_pred, v_pred, p_pred = predict(model, coords_t)
    u_pred = u_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)
    v_pred = v_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)
    p_pred = p_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)

    # L2 relative error vs solver
    u_true = fr["u"]
    v_true = fr["v"]
    l2_u = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
    l2_v = np.linalg.norm(v_pred - v_true) / np.linalg.norm(v_true)
    l2_p = np.linalg.norm(p_pred - fr["p"]) / np.linalg.norm(fr["p"])
    print(f"\nL2 relative error vs LBM:")
    print(f"  u: {l2_u:.4f}  v: {l2_v:.4f}  p: {l2_p:.4f}")

    # --- Save results as numpy for plotting ------------------------------
    np.savez(os.path.join(case_dir, "prediction_steady.npz"),
             u_pred=u_pred, v_pred=v_pred, p_pred=p_pred,
             u_true=u_true, v_true=v_true, p_true=fr["p"],
             Xn=Xn, Yn=Yn, obstacle=obstacle,
             l2_u=l2_u, l2_v=l2_v, l2_p=l2_p)
    print(f"Prediction saved: {case_dir}/prediction_steady.npz")

    # --- Loss history for plotting ----------------------------------------
    np.savez(os.path.join(case_dir, "loss_history.npz"), **history)
    print(f"Loss history saved: {case_dir}/loss_history.npz")

    # --- Summary ---------------------------------------------------------
    print(f"\n=== SUMMARY ===")
    print(f"Case:    cylinder Re={cfg.re}")
    print(f"Device:  {device}")
    print(f"Params:  {n_params}")
    print(f"Time:    {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"L2 u:    {l2_u:.4f}")
    print(f"L2 v:    {l2_v:.4f}")
    print(f"L2 p:    {l2_p:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="cylinder_re100")
    parser.add_argument("--epochs-adam", type=int, default=EPOCHS_ADAM)
    parser.add_argument("--epochs-lbfgs", type=int, default=EPOCHS_LBFGS)
    parser.add_argument("--n-colloc", type=int, default=N_COLLOC)
    parser.add_argument("--n-sensors", type=int, default=N_SENSORS)
    args = parser.parse_args()
    EPOCHS_ADAM = args.epochs_adam
    EPOCHS_LBFGS = args.epochs_lbfgs
    N_COLLOC = args.n_colloc
    N_SENSORS = args.n_sensors
    train(args)
