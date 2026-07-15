"""Train parametric PINN on Lid-Driven Cavity (Re=100, 400).

Usage:
    python3 train_cavity.py                        # multi-Re (Re=100 + Re=400)
    python3 train_cavity.py --single-re 100        # single Re only
    python3 train_cavity.py --single-re 400        # single Re only
    python3 train_cavity.py --epochs-adam 8000     # longer Adam phase
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cavity_re100, cavity_re400, CaseConfig
from data.loader import (
    load_frame, grid_coords, flatten_grid,
    make_collocation, subsample_sensors,
)
from models.pinn import ParametricPINN, predict
from models.losses import (
    total_loss_cavity, total_loss_cavity_multi_re,
    pde_loss, data_loss, bc_loss_cavity,
)

# ---------------------------------------------------------------------------
# Defaults (improved for higher accuracy)
# ---------------------------------------------------------------------------
N_COLLOC    = 10000     # doubled: denser PDE supervision
N_SENSORS   = 3000      # tripled: 18% grid coverage (was 6%)
LR_ADAM     = 5e-4      # lower initial LR for stability
EPOCHS_ADAM = 15000     # 3x longer Adam phase
EPOCHS_LBFGS = 1000     # 2x longer L-BFGS
W_PDE       = 5.0       # PDE was underweighted vs data/bc; rebalanced
W_DATA      = 5.0       # stronger data supervision, capped to avoid overfit
W_BC        = 5.0       # stronger BC enforcement
U_LID       = 0.1
FRAME_NAME  = "frame_12800.json"
HIDDEN      = 256       # 2x wider network
N_LAYERS    = 8
N_FREQS     = 128       # Fourier feature frequencies (lifts x,y to 512-dim)
SIGMA       = 5.0       # Fourier feature frequency scale
GRAD_CLIP   = 1.0       # gradient clipping to tame 2nd-deriv scaling

# Re normalization: map [100, 1000] -> [0, 1]
RE_MIN = 100.0
RE_MAX = 1000.0


def normalize_re(re: float) -> float:
    """Normalize Re to [0, 1] range."""
    return (re - RE_MIN) / (RE_MAX - RE_MIN)


def to_device(t, device):
    return t.to(device, dtype=torch.float32)


def prepare_single_re(cfg: CaseConfig, device, importance_sample=True):
    """Load data for a single Re case and prepare tensors."""
    fr = load_frame(os.path.join(cfg.case_dir, "frames", FRAME_NAME))
    Xn, Yn, _, _ = grid_coords(cfg)
    coords_all = flatten_grid(Xn, Yn)
    obstacle = fr["obstacle"]

    colloc = make_collocation(cfg, N_COLLOC, seed=0, obstacle=obstacle)
    sens = subsample_sensors(
        fr["u"], fr["v"], fr["p"], obstacle, N_SENSORS, seed=2,
        normalized_coords=coords_all,
        importance_sample=importance_sample,
    )

    return {
        "cfg": cfg,
        "fr": fr,
        "coords_all": coords_all,
        "colloc": colloc,
        "sens": sens,
        "obstacle": obstacle,
    }


def train_single_re(args):
    """Train parametric PINN at a single Re value."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    if args.single_re == 100:
        cfg = cavity_re100()
    elif args.single_re == 400:
        cfg = cavity_re400()
    else:
        raise ValueError(f"Unsupported Re={args.single_re}. Use 100 or 400.")

    case_dir = os.path.join(cfg.case_dir, "pinn_parametric")
    os.makedirs(case_dir, exist_ok=True)

    data = prepare_single_re(cfg, device)
    colloc_t = to_device(torch.from_numpy(data["colloc"]), device)
    sens_coords_t = to_device(torch.from_numpy(data["sens"]["coords"]), device)
    sens_u_t = to_device(torch.from_numpy(data["sens"]["u"]), device)
    sens_v_t = to_device(torch.from_numpy(data["sens"]["v"]), device)

    re_norm = normalize_re(cfg.re)
    print(f"Cavity Re={cfg.re} (Re_norm={re_norm:.3f})")
    print(f"Collocation: {data['colloc'].shape}  Sensors: {data['sens']['coords'].shape}")

    model = ParametricPINN(n_params=1, hidden=HIDDEN, n_layers=N_LAYERS,
                           n_freqs=N_FREQS, sigma=SIGMA).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, factor=0.5, patience=500, min_lr=1e-5
    )

    # Augment collocation and sensor coords with Re parameter
    re_colloc = np.full((data["colloc"].shape[0], 1), re_norm)
    colloc_aug = np.concatenate([data["colloc"], re_colloc], axis=1)
    colloc_aug_t = to_device(torch.from_numpy(colloc_aug), device)

    re_sens = np.full((data["sens"]["coords"].shape[0], 1), re_norm)
    sens_aug = np.concatenate([data["sens"]["coords"], re_sens], axis=1)
    sens_aug_t = to_device(torch.from_numpy(sens_aug), device)

    t0 = time.time()
    history = {"epoch": [], "loss": [], "pde": [], "data": [], "bc": []}

    print(f"\n--- Adam training ({EPOCHS_ADAM} epochs) ---")
    for ep in range(1, EPOCHS_ADAM + 1):
        optimizer_adam.zero_grad()
        L, Lpde, Ldata, Lbc = total_loss_cavity(
            model, colloc_aug_t, sens_aug_t, sens_u_t, sens_v_t,
            re=cfg.re, u_lid=U_LID, device=device,
            w_pde=W_PDE, w_data=W_DATA, w_bc=W_BC,
            re_norm=re_norm,
        )
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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

    # --- L-BFGS fine-tune ---
    print(f"\n--- L-BFGS fine-tune ({EPOCHS_LBFGS} steps) ---")
    t1 = time.time()
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0,
        max_iter=EPOCHS_LBFGS, history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        L, _, _, _ = total_loss_cavity(
            model, colloc_aug_t, sens_aug_t, sens_u_t, sens_v_t,
            re=cfg.re, u_lid=U_LID, device=device,
            w_pde=W_PDE, w_data=W_DATA, w_bc=W_BC,
            re_norm=re_norm,
        )
        L.backward()
        return L

    L_lbfgs = optimizer_lbfgs.step(closure)
    lbfs_time = time.time() - t1
    print(f"L-BFGS done in {lbfs_time:.1f}s  final loss={L_lbfgs.item():.6f}")

    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # --- Save model ---
    model_path = os.path.join(case_dir, "model_parametric.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "n_params": model.n_params,
        "n_freqs": model.n_freqs,
        "sigma": model.sigma,
        "hidden": HIDDEN,
        "n_layers": N_LAYERS,
    }, model_path)
    print(f"Model saved: {model_path}")

    # --- Inference on full grid ---
    model.eval()
    re_colloc_full = np.full((data["coords_all"].shape[0], 1), re_norm)
    coords_aug = np.concatenate([data["coords_all"], re_colloc_full], axis=1)
    coords_t = to_device(torch.from_numpy(coords_aug), device)
    with torch.no_grad():
        u_pred, v_pred, p_pred = predict(model, coords_t)
    u_pred = u_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)
    v_pred = v_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)
    p_pred = p_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)

    u_true = data["fr"]["u"]
    v_true = data["fr"]["v"]
    p_true = data["fr"]["p"]
    l2_u = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
    l2_v = np.linalg.norm(v_pred - v_true) / np.linalg.norm(v_true)
    l2_p = np.linalg.norm(p_pred - p_true) / np.linalg.norm(p_true)
    print(f"\nL2 relative error vs LBM:")
    print(f"  u: {l2_u:.4f}  v: {l2_v:.4f}  p: {l2_p:.4f}")

    np.savez(os.path.join(case_dir, "prediction_parametric.npz"),
             u_pred=u_pred, v_pred=v_pred, p_pred=p_pred,
             u_true=u_true, v_true=v_true, p_true=p_true,
             obstacle=data["obstacle"],
             l2_u=l2_u, l2_v=l2_v, l2_p=l2_p,
             re=cfg.re, re_norm=re_norm)
    print(f"Prediction saved: {case_dir}/prediction_parametric.npz")

    np.savez(os.path.join(case_dir, "loss_history.npz"), **history)
    print(f"Loss history saved: {case_dir}/loss_history.npz")

    print(f"\n=== SUMMARY ===")
    print(f"Case:    Cavity Re={cfg.re}")
    print(f"Device:  {device}")
    print(f"Params:  {n_params}")
    print(f"Time:    {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"L2 u:    {l2_u:.4f}")
    print(f"L2 v:    {l2_v:.4f}")
    print(f"L2 p:    {l2_p:.4f}")


def train_multi_re(args):
    """Train parametric PINN across multiple Re values (Re=100 + Re=400)."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    configs = [cavity_re100(), cavity_re400()]
    output_dir = os.path.join(configs[0].case_dir, "..", "pinn_parametric")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for each Re
    dataset = {}
    for cfg in configs:
        data = prepare_single_re(cfg, device)
        re_norm = normalize_re(cfg.re)
        dataset[cfg.re] = {
            "cfg": cfg,
            "data": data,
            "re_norm": re_norm,
        }
        print(f"Re={cfg.re} (norm={re_norm:.3f}): colloc={data['colloc'].shape}  "
              f"sensors={data['sens']['coords'].shape}")

    model = ParametricPINN(n_params=1, hidden=HIDDEN, n_layers=N_LAYERS,
                           n_freqs=N_FREQS, sigma=SIGMA).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, factor=0.5, patience=500, min_lr=1e-5
    )

    # Build augmented tensors for each Re
    colloc_by_re = {}
    sens_by_re = {}
    for re_val, entry in dataset.items():
        cfg = entry["cfg"]
        data = entry["data"]
        re_norm = entry["re_norm"]

        re_colloc = np.full((data["colloc"].shape[0], 1), re_norm)
        colloc_by_re[re_val] = np.concatenate([data["colloc"], re_colloc], axis=1)

        re_sens = np.full((data["sens"]["coords"].shape[0], 1), re_norm)
        sens_by_re[re_val] = {
            "coords": np.concatenate([data["sens"]["coords"], re_sens], axis=1),
            "u": data["sens"]["u"],
            "v": data["sens"]["v"],
            "p": data["sens"]["p"],
        }

    t0 = time.time()
    history = {"epoch": [], "loss": [], "pde": [], "data": [], "bc": []}

    print(f"\n--- Adam training ({EPOCHS_ADAM} epochs) across {len(configs)} Re values ---")
    for ep in range(1, EPOCHS_ADAM + 1):
        optimizer_adam.zero_grad()
        L, Lpde, Ldata, Lbc = total_loss_cavity_multi_re(
            model, colloc_by_re, sens_by_re, u_lid=U_LID,
            device=device, w_pde=W_PDE, w_data=W_DATA, w_bc=W_BC,
        )
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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

    # --- L-BFGS fine-tune ---
    print(f"\n--- L-BFGS fine-tune ({EPOCHS_LBFGS} steps) ---")
    t1 = time.time()
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0,
        max_iter=EPOCHS_LBFGS, history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        L, _, _, _ = total_loss_cavity_multi_re(
            model, colloc_by_re, sens_by_re, u_lid=U_LID,
            device=device, w_pde=W_PDE, w_data=W_DATA, w_bc=W_BC,
        )
        L.backward()
        return L

    L_lbfgs = optimizer_lbfgs.step(closure)
    lbfs_time = time.time() - t1
    print(f"L-BFGS done in {lbfs_time:.1f}s  final loss={L_lbfgs.item():.6f}")

    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # --- Save model ---
    model_path = os.path.join(output_dir, "model_cavity_multi_re.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "n_params": model.n_params,
        "n_freqs": model.n_freqs,
        "sigma": model.sigma,
        "hidden": HIDDEN,
        "n_layers": N_LAYERS,
    }, model_path)
    print(f"Model saved: {model_path}")

    # --- Inference and L2 error for each Re ---
    model.eval()
    for re_val, entry in dataset.items():
        cfg = entry["cfg"]
        data = entry["data"]
        re_norm = entry["re_norm"]

        re_colloc_full = np.full((data["coords_all"].shape[0], 1), re_norm)
        coords_aug = np.concatenate([data["coords_all"], re_colloc_full], axis=1)
        coords_t = to_device(torch.from_numpy(coords_aug), device)
        with torch.no_grad():
            u_pred, v_pred, p_pred = predict(model, coords_t)
        u_pred_r = u_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)
        v_pred_r = v_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)
        p_pred_r = p_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)

        u_true = data["fr"]["u"]
        v_true = data["fr"]["v"]
        p_true = data["fr"]["p"]
        l2_u = np.linalg.norm(u_pred_r - u_true) / np.linalg.norm(u_true)
        l2_v = np.linalg.norm(v_pred_r - v_true) / np.linalg.norm(v_true)
        l2_p = np.linalg.norm(p_pred_r - p_true) / np.linalg.norm(p_true)
        print(f"\nRe={re_val} L2 error: u={l2_u:.4f}  v={l2_v:.4f}  p={l2_p:.4f}")

        np.savez(os.path.join(output_dir, f"prediction_re{int(re_val)}.npz"),
                 u_pred=u_pred_r, v_pred=v_pred_r, p_pred=p_pred_r,
                 u_true=u_true, v_true=v_true, p_true=p_true,
                 obstacle=data["obstacle"],
                 l2_u=l2_u, l2_v=l2_v, l2_p=l2_p,
                 re=re_val, re_norm=re_norm)

    np.savez(os.path.join(output_dir, "loss_history_multi_re.npz"), **history)
    print(f"\nLoss history saved: {output_dir}/loss_history_multi_re.npz")

    print(f"\n=== SUMMARY (Multi-Re) ===")
    print(f"Cases:   Cavity Re=100, 400")
    print(f"Device:  {device}")
    print(f"Params:  {n_params}")
    print(f"Time:    {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-re", type=int, default=None,
                        help="Train at single Re (100 or 400). If None, multi-Re.")
    parser.add_argument("--epochs-adam", type=int, default=EPOCHS_ADAM)
    parser.add_argument("--epochs-lbfgs", type=int, default=EPOCHS_LBFGS)
    parser.add_argument("--n-colloc", type=int, default=N_COLLOC)
    parser.add_argument("--n-sensors", type=int, default=N_SENSORS)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--lr", type=float, default=LR_ADAM)
    args = parser.parse_args()
    EPOCHS_ADAM = args.epochs_adam
    EPOCHS_LBFGS = args.epochs_lbfgs
    N_COLLOC = args.n_colloc
    N_SENSORS = args.n_sensors
    HIDDEN = args.hidden
    LR_ADAM = args.lr

    if args.single_re is not None:
        train_single_re(args)
    else:
        train_multi_re(args)
