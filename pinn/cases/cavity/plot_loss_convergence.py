"""Plot training loss convergence for the cavity temporal PINN (Phase 6.8).

Left panel: total loss vs epoch (log scale), annotated with the reduction factor.
Right panel: individual components (PDE, data, BC, IC) vs epoch.

Output: docs/assets/images/cavity/loss_convergence.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NPZ = os.path.join(PROJECT_ROOT, "output", "cavity", "pinn", "temporal", "v1", "loss_history.npz")
OUT = os.path.join(PROJECT_ROOT, "docs", "assets", "images", "cavity", "loss_convergence.png")

# Dracula-inspired theme
BG = "#0d1117"
FG = "#c9d1d9"
ACCENT = "#58a6ff"   # cyan/blue
PANEL = "#161b22"
GRID = "#30363d"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "text.color": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "font.family": "monospace",
    "font.size": 10,
})


def main():
    d = np.load(NPZ)
    epoch = d["epoch"]
    loss = d["loss"]
    pde = d["pde"]
    data = d["data"]
    bc = d["bc"]
    ic = d["ic"]

    reduction = loss[0] / loss[-1]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: total loss (log scale)
    ax_l.semilogy(epoch, loss, color=ACCENT, lw=2, label="total loss")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Total loss (log)")
    ax_l.set_title("Training Loss Convergence")
    ax_l.grid(True, color=GRID, alpha=0.4)
    ax_l.annotate(
        f"{reduction:.1e}x reduction",
        xy=(epoch[-1], loss[-1]),
        xytext=(epoch[int(len(epoch)*0.35)], loss[int(len(epoch)*0.45)]),
        color=FG, fontsize=10,
        arrowprops=dict(color=GRID, arrowstyle="->"))

    # Right: components
    ax_r.semilogy(epoch, pde, color="#ff7b72", lw=1.5, label="PDE (NS residual)")
    ax_r.semilogy(epoch, data, color="#3fb950", lw=1.5, label="Data (sensors)")
    ax_r.semilogy(epoch, bc, color="#d2a8ff", lw=1.5, label="BC (walls + lid)")
    ax_r.semilogy(epoch, ic, color="#e3b341", lw=1.5, label="IC (rest state)")
    ax_r.set_xlabel("Epoch")
    ax_r.set_ylabel("Component loss (log)")
    ax_r.set_title("Loss Components")
    ax_r.grid(True, color=GRID, alpha=0.4)
    ax_r.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"Total loss reduction: {reduction:.2e}x ({loss[0]:.4f} -> {loss[-1]:.6f})")


if __name__ == "__main__":
    main()
