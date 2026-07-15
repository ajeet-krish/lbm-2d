"""Evaluate a trained PINN on the full grid and produce inference output.

Usage:
    python3 evaluate.py                              # default: cylinder_re100
    python3 evaluate.py --case cylinder_re100
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CaseConfig, cylinder_re100
from data.loader import load_frame, grid_coords, flatten_grid
from models.pinn import PINN, predict


def evaluate(cfg: CaseConfig, model_path: str, output_path: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    fr = load_frame(os.path.join(cfg.case_dir, "frames", "frame_18000.json"))
    Xn, Yn, _, _ = grid_coords(cfg)
    coords_all = flatten_grid(Xn, Yn)

    model = PINN(hidden=64, n_layers=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    coords_t = torch.from_numpy(coords_all).float().to(device)
    with torch.no_grad():
        u_pred, v_pred, p_pred = predict(model, coords_t)

    u_pred = u_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)
    v_pred = v_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)
    p_pred = p_pred.cpu().numpy().reshape(cfg.ny, cfg.nx)

    l2_u = np.linalg.norm(u_pred - fr["u"]) / np.linalg.norm(fr["u"])
    l2_v = np.linalg.norm(v_pred - fr["v"]) / np.linalg.norm(fr["v"])
    l2_p = np.linalg.norm(p_pred - fr["p"]) / np.linalg.norm(fr["p"])
    print(f"L2 relative error:  u={l2_u:.4f}  v={l2_v:.4f}  p={l2_p:.4f}")

    np.savez(output_path,
             u_pred=u_pred, v_pred=v_pred, p_pred=p_pred,
             u_true=fr["u"], v_true=fr["v"], p_true=fr["p"],
             Xn=Xn, Yn=Yn, obstacle=fr["obstacle"],
             l2_u=l2_u, l2_v=l2_v, l2_p=l2_p)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="cylinder_re100")
    args = parser.parse_args()

    cfg = cylinder_re100()
    case_dir = os.path.join(cfg.case_dir, "pinn")
    model_path = os.path.join(case_dir, "model_steady.pt")
    output_path = os.path.join(case_dir, "prediction.npz")
    evaluate(cfg, model_path, output_path)
