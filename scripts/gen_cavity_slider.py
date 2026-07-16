"""Regenerate the cavity Velocity Field comparison-slider images.

The plotted FIELD (the cavity domain) is square (equal aspect), but the
image itself may be non-square: it keeps the black obstacle (wall) overlay
and a velocity-magnitude colorbar with a text label, exactly like the
solver's standard field plots.

  * re{re}_contour.png     -- velocity-magnitude contour (viridis) + walls + cbar
  * re{re}_streamlines.png  -- velocity-magnitude streamlines (viridis) + walls + cbar

This mirrors the square LBM/PINN animation in colormap and field shape; the
colorbar just makes the file slightly wider than tall.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(ROOT, "docs", "assets", "images", "cavity")
FRAME = "frame_10240.json"          # steady-state frame
CASES = [("re100", 100), ("re400", 400)]
CMAP = "jet"


def load(path):
    with open(path) as f:
        d = json.load(f)
    nx, ny = int(d["nx"]), int(d["ny"])
    u = np.asarray(d["u"], dtype=float).reshape(ny, nx)
    v = np.asarray(d["v"], dtype=float).reshape(ny, nx)
    obs = np.asarray(d.get("obstacle", []), dtype=bool).reshape(ny, nx) \
        if "obstacle" in d else None
    return u, v, obs


def save(path, fig):
    # Trim whitespace so the field stays square and the colorbar sits at the
    # right edge (image is slightly wider than tall -- that's expected).
    fig.savefig(path, dpi=110, facecolor="white", edgecolor="none",
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    for sub, re in CASES:
        u, v, obs = load(os.path.join(ROOT, "output", "cavity", sub, "frames", FRAME))
        vel = np.hypot(u, v)
        vmax = float(max(vel.max(), 1e-3))

        # ---- Contour: square field + black walls + velocity colorbar ----
        fig, ax = plt.subplots(figsize=(7.4, 6.6))
        im = ax.imshow(vel, origin="lower", cmap=CMAP, vmin=0.0, vmax=vmax,
                       interpolation="bilinear", aspect="equal")
        ax.set_axis_off()
        if obs is not None:
            mask = np.ma.masked_where(~obs, np.ones_like(obs, float))
            ax.imshow(mask, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                     alpha=1.0, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("Velocity Magnitude (m/s)", color="black")
        save(os.path.join(IMG_DIR, f"re{re}_contour.png"), fig)

        # ---- Streamlines: square field + black walls + velocity colorbar ----
        fig, ax = plt.subplots(figsize=(7.4, 6.6))
        ny, nx = u.shape
        yg, xg = np.mgrid[0:ny, 0:nx]
        step = max(1, nx // 48)
        smag = vel[::step, ::step]
        sp = ax.streamplot(xg[::step, ::step], yg[::step, ::step],
                          u[::step, ::step], v[::step, ::step],
                          color=smag, cmap=CMAP, density=1.1,
                          linewidth=0.8, arrowsize=0.8)
        ax.set_axis_off()
        if obs is not None:
            mask = np.ma.masked_where(~obs, np.ones_like(obs, float))
            ax.imshow(mask, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                     alpha=1.0, interpolation="nearest")
        cbar = fig.colorbar(sp.lines, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("Velocity Magnitude (m/s)", color="black")
        save(os.path.join(IMG_DIR, f"re{re}_streamlines.png"), fig)
        print(f"  re{re}: contour + streamlines saved (vmax={vmax:.4f})")


if __name__ == "__main__":
    main()
