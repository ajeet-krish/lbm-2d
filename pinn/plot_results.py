"""Generate 3-panel comparison plot: LBM / PINN / Error delta.

Also generates loss history plot. All matplotlib (no torch).

Usage:
    python3 plot_results.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cylinder_re100

CASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "output", "cylinder", "re100", "pinn"
)


def plot_comparison(case_dir=CASE_DIR):
    """3-panel: LBM / PINN / absolute error delta."""
    data = np.load(os.path.join(case_dir, "prediction_steady.npz"))
    u_true = data["u_true"]
    u_pred = data["u_pred"]
    v_true = data["v_true"]
    v_pred = data["v_pred"]
    obstacle = data["obstacle"]
    l2_u = float(data["l2_u"])
    l2_v = float(data["l2_v"])

    ny, nx = u_true.shape
    # Velocity magnitude
    vel_true = np.sqrt(u_true**2 + v_true**2)
    vel_pred = np.sqrt(u_pred**2 + v_pred**2)
    err = np.abs(vel_true - vel_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    vmin = 0.0
    vmax = max(vel_true.max(), vel_pred.max())

    # Panel A: C++ LBM Solver
    im0 = axes[0].imshow(vel_true, origin="lower", cmap="coolwarm",
                          vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("C++ LBM Solver (Baseline)", fontsize=11)
    axes[0].set_xlabel("x (downsampled)")
    axes[0].set_ylabel("y (downsampled)")

    # Panel B: PINN Surrogate
    im1 = axes[1].imshow(vel_pred, origin="lower", cmap="coolwarm",
                          vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title(f"PINN Surrogate  (L2u={l2_u:.1%})", fontsize=11)
    axes[1].set_xlabel("x (downsampled)")

    # Panel C: Error Delta
    im2 = axes[2].imshow(err, origin="lower", cmap="Reds",
                          vmin=0.0, vmax=vmax * 0.5, aspect="auto")
    axes[2].set_title(f"|Error|  (L2v={l2_v:.1%})", fontsize=11)
    axes[2].set_xlabel("x (downsampled)")

    plt.colorbar(im0, ax=axes[0], shrink=0.8, label="|V|")
    plt.colorbar(im1, ax=axes[1], shrink=0.8, label="|V|")
    plt.colorbar(im2, ax=axes[2], shrink=0.8, label="|V| error")

    fig.suptitle("Cylinder Re=100: Steady-State PINN vs LBM",
                 fontsize=13, fontweight="bold")
    out = os.path.join(case_dir, "comparison_steady.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_loss_history(case_dir=CASE_DIR):
    """Loss convergence curves."""
    data = np.load(os.path.join(case_dir, "loss_history.npz"))
    epochs = data["epoch"]
    loss = data["loss"]
    pde = data["pde"]
    data_l = data["data"]
    bc = data["bc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, loss, label="Total loss", linewidth=2)
    ax.semilogy(epochs, pde, label="PDE residual", linewidth=1.5, alpha=0.8)
    ax.semilogy(epochs, data_l, label="Data (sensor fit)", linewidth=1.5, alpha=0.8)
    ax.semilogy(epochs, bc, label="Boundary conditions", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("PINN Training Convergence -- Cylinder Re=100")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(case_dir, "loss_history.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    plot_comparison()
    plot_loss_history()
