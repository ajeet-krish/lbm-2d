"""Plot frame-by-frame temporal L2 error for the cavity temporal PINN (Phase 6.8).

Compares pinn_temporal_re{re}.bin against lbm_re{re}.bin frame-by-frame.

X-axis: frame number (0-50). Y-axis: L2 error (%).
Two lines per panel (u-L2, v-L2), two panels (Re=100, Re=400).

Output: docs/assets/images/cavity/temporal_l2_profile.png
"""

import os
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "docs", "assets", "data", "cavity")
OUT = os.path.join(PROJECT_ROOT, "docs", "assets", "images", "cavity", "temporal_l2_profile.png")

MAGIC = 0x4C424D31

# Dracula-inspired theme
BG = "#0d1117"
FG = "#c9d1d9"
PANEL = "#161b22"
GRID = "#30363d"


def parse_binary(path):
    with open(path, "rb") as f:
        magic, n_frames, nx, ny, n_chan, dtype_flag = struct.unpack("<IIIIII", f.read(24))
        assert magic == MAGIC, f"bad magic {magic:#x}"
        raw = np.frombuffer(f.read(), dtype=np.uint16)
        arr = raw.view(np.float16).astype(np.float32)
        arr = arr.reshape(n_frames, n_chan, ny, nx)
    return arr  # (n_frames, n_chan, ny, nx) : [0]=u, [1]=v, [2]=obstacle


def frame_l2(pred_u, pred_v, true_u, true_v, obstacle):
    # LBM binary stores obstacle=1 for fluid, 0 for wall (inverted vs raw JSON).
    fluid = obstacle > 0.5
    def l2(p, t):
        p_f = p[fluid]
        t_f = t[fluid]
        denom = np.sqrt(np.mean(t_f**2)) + 1e-12
        return 100.0 * np.sqrt(np.mean((p_f - t_f)**2)) / denom
    return l2(pred_u, true_u), l2(pred_v, true_v)


def main():
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, re in zip(axes, [100, 400]):
        pinn = parse_binary(os.path.join(DATA_DIR, f"pinn_temporal_re{re}.bin"))
        lbm = parse_binary(os.path.join(DATA_DIR, f"lbm_re{re}.bin"))
        n = min(pinn.shape[0], lbm.shape[0])
        u_err, v_err = [], []
        for fi in range(n):
            pu, pv = pinn[fi, 0], pinn[fi, 1]
            tu, tv = lbm[fi, 0], lbm[fi, 1]
            obs = lbm[fi, 2]
            eu, ev = frame_l2(pu, pv, tu, tv, obs)
            u_err.append(eu)
            v_err.append(ev)
        frames = np.arange(n)
        # Frame 0 is the rest state (true u=v=0); exclude from mean to avoid
        # division by zero in the relative metric.
        u_mean = np.mean(u_err[1:]) if n > 1 else np.mean(u_err)
        v_mean = np.mean(v_err[1:]) if n > 1 else np.mean(v_err)
        ax.plot(frames, u_err, color="#58a6ff", lw=2, label="u-L2 (%)")
        ax.plot(frames, v_err, color="#ff7b72", lw=2, label="v-L2 (%)")
        ax.axvline(0, color=GRID, ls="--", alpha=0.5)
        ax.set_xlabel("Frame")
        ax.set_title(f"Re = {re}")
        ax.grid(True, color=GRID, alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")
        # Annotate early transient
        ax.axvspan(0, 10, color=GRID, alpha=0.15)
        ax.annotate(
            "early transient\nhardest (~45%)",
            xy=(5, max(u_err[:11])),
            xytext=(12, max(u_err[:11]) * 0.8),
            color=FG, fontsize=8,
            arrowprops=dict(color=GRID, arrowstyle="->"))

    axes[0].set_ylabel("L2 error (%)")
    fig.suptitle("Frame-by-Frame Temporal L2 Error (PINN vs LBM)", color=FG, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}")

    # Print summary
    for re in [100, 400]:
        pinn = parse_binary(os.path.join(DATA_DIR, f"pinn_temporal_re{re}.bin"))
        lbm = parse_binary(os.path.join(DATA_DIR, f"lbm_re{re}.bin"))
        n = min(pinn.shape[0], lbm.shape[0])
        u_e, v_e = [], []
        for fi in range(n):
            pu, pv = pinn[fi, 0], pinn[fi, 1]
            tu, tv = lbm[fi, 0], lbm[fi, 1]
            obs = lbm[fi, 2]
            eu, ev = frame_l2(pu, pv, tu, tv, obs)
            u_e.append(eu); v_e.append(ev)
        u_mean = np.mean(u_e[1:]); v_mean = np.mean(v_e[1:])
        print(f"Re={re}: u-L2 mean={u_mean:.1f}% (excl. frame0) final={u_e[-1]:.1f}%  "
              f"v-L2 mean={v_mean:.1f}% final={v_e[-1]:.1f}%")


if __name__ == "__main__":
    main()
