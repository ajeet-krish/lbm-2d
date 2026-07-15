"""Data loading and sampling utilities for the PINN surrogate suite.

torch-free (numpy only). Reads the C++ solver's JSON frame output and produces
the point sets a PINN needs:

  * frame fields (u, v, p, omega, velocity, obstacle) as 2D numpy arrays
  * a full normalized coordinate grid for inference / plotting
  * random interior collocation points (for the PDE residual loss)
  * sparse "sensor" subsamples of the solver field (for the data loss)
  * geometry-aware boundary points (e.g. the cylinder circumference)

All coordinates returned by this module are normalized to [-1, 1] via
CaseConfig.normalize(), matching the convention used by the network.
"""

import json
import os
import sys
from typing import Tuple

import numpy as np

# Make the pinn/ package root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CaseConfig


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_meta(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_frame(path: str) -> dict:
    """Load one frame_*.json into a dict of 2D numpy arrays (ny, nx)."""
    with open(path) as f:
        d = json.load(f)
    nx = int(d["nx"])
    ny = int(d["ny"])
    out = {"nx": nx, "ny": ny}
    for key in ("velocity", "u", "v", "rho", "p", "omega", "obstacle"):
        if key in d:
            arr = np.asarray(d[key], dtype=np.float64).reshape(ny, nx)
            out[key] = arr
    return out


def load_forces(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read forces.jsonl -> (steps, cd, cl) 1D arrays."""
    steps, cd, cl = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            steps.append(rec["step"])
            cd.append(rec["cd"])
            cl.append(rec["cl"])
    return (np.asarray(steps), np.asarray(cd), np.asarray(cl))


# --------------------------------------------------------------------------
# Coordinate grids
# --------------------------------------------------------------------------
def grid_coords(cfg: CaseConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the full normalized (Xn, Yn) and physical (X, Y) meshes.

    Shapes are (ny, nx). Physical coords use (x = i*ds, y = j*ds).
    """
    i = np.arange(cfg.nx)
    j = np.arange(cfg.ny)
    I, J = np.meshgrid(i, j, indexing="ij")
    X, Y = cfg.physical_coords(I, J)
    Xn, Yn = cfg.normalize(X, Y)
    return Xn, Yn, X, Y


def flatten_grid(Xn: np.ndarray, Yn: np.ndarray) -> np.ndarray:
    """Stack normalized meshes into an (N, 2) coordinate array."""
    return np.stack([Xn.ravel(), Yn.ravel()], axis=1)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def make_collocation(cfg: CaseConfig, n: int, seed: int = 0,
                     obstacle: np.ndarray = None) -> np.ndarray:
    """Random interior collocation points in [-1, 1]^2 (excluding obstacle).

    The PDE residual is evaluated at these points. For the cylinder we reject
    points inside the circle. For cavities and other enclosed domains, we
    reject points on obstacle nodes.
    """
    rng = np.random.default_rng(seed)
    pts = []
    geom = cfg.geometry
    inside_obstacle = None
    if obstacle is not None:
        inside_obstacle = (obstacle > 0.5)
    while len(pts) < n:
        need = n - len(pts) + 64
        xn = rng.uniform(-1.0, 1.0, size=need)
        yn = rng.uniform(-1.0, 1.0, size=need)
        X, Y = cfg.denormalize(xn, yn)
        if "radius" in geom:
            cx, cy = cfg.physical_coords(geom["cx"], geom["cy"])
            r = geom["radius"] * cfg.ds
            dist2 = (X - cx) ** 2 + (Y - cy) ** 2
            keep = dist2 > (r * 1.05) ** 2
        else:
            keep = np.ones_like(X, dtype=bool)
        if inside_obstacle is not None:
            ii = np.rint(X / cfg.ds).astype(int)
            jj = np.rint(Y / cfg.ds).astype(int)
            ii = np.clip(ii, 0, cfg.nx - 1)
            jj = np.clip(jj, 0, cfg.ny - 1)
            on_obst = inside_obstacle[jj, ii]
            keep = keep & (~on_obst)
        sel = np.stack([xn[keep], yn[keep]], axis=1)
        pts.append(sel)
    return np.concatenate(pts, axis=0)[:n]


def make_boundary(cfg: CaseConfig, n: int, seed: int = 1) -> np.ndarray:
    """Geometry boundary points in normalized space.

    Cylinder: points on the circumference x^2 + y^2 = R^2.
    Cavity: points along the four walls (bottom, left, right, top).
    """
    geom = cfg.geometry
    if "radius" in geom:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        i = geom["cx"] + geom["radius"] * np.cos(theta)
        j = geom["cy"] + geom["radius"] * np.sin(theta)
        X, Y = cfg.physical_coords(i, j)
        Xn, Yn = cfg.normalize(X, Y)
        return np.stack([Xn, Yn], axis=1)
    if geom.get("shape_type") == "lid-driven-cavity":
        # Distribute n points evenly along the four walls.
        pts_per_wall = n // 4
        rng = np.random.default_rng(seed)
        # Bottom wall: y=0, x in [0, NX]
        x_bot = rng.uniform(0, cfg.NX, size=pts_per_wall)
        y_bot = np.zeros_like(x_bot)
        # Top wall: y=NY, x in [0, NX]
        x_top = rng.uniform(0, cfg.NX, size=pts_per_wall)
        y_top = np.full_like(x_top, cfg.NY)
        # Left wall: x=0, y in [0, NY]
        x_left = np.zeros(pts_per_wall)
        y_left = rng.uniform(0, cfg.NY, size=pts_per_wall)
        # Right wall: x=NX, y in [0, NY]
        x_right = np.full(pts_per_wall, cfg.NX)
        y_right = rng.uniform(0, cfg.NY, size=pts_per_wall)
        X = np.concatenate([x_bot, x_top, x_left, x_right])
        Y = np.concatenate([y_bot, y_top, y_left, y_right])
        Xn, Yn = cfg.normalize(X, Y)
        return np.stack([Xn, Yn], axis=1)
    raise NotImplementedError(
        f"Boundary generation not implemented for shape {cfg.shape_type}")


def subsample_sensors(u: np.ndarray, v: np.ndarray, p: np.ndarray,
                      obstacle: np.ndarray, n: int, seed: int = 2,
                      normalized_coords: np.ndarray = None,
                      importance_sample: bool = False) -> dict:
    """Randomly subsample fluid nodes as sparse sensor measurements.

    Returns a dict with 'coords' (N, 2) normalized, and targets 'u', 'v', 'p'
    (N,) sampled from the solver field. Simulates sparse wind-tunnel style
    measurements; the PDE residual fills the gaps between sensors.

    If importance_sample=True, ~50% of sensors are placed near high-gradient
    regions (vortex core, walls, boundary layers) using velocity magnitude
    gradient as a weighting function.
    """
    rng = np.random.default_rng(seed)
    fluid = (obstacle < 0.5)
    idx = np.argwhere(fluid)

    if importance_sample and idx.shape[0] > n:
        # Compute velocity magnitude gradient as importance weight
        vmag = np.sqrt(u**2 + v**2)
        from scipy.ndimage import gaussian_filter
        vmag_smooth = gaussian_filter(vmag, sigma=2.0)
        grad_x = np.gradient(vmag_smooth, axis=1)
        grad_y = np.gradient(vmag_smooth, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Weight: high gradient -> high weight. Flatten obstacle to 0.
        weight_map = grad_mag.copy()
        weight_map[obstacle > 0.5] = 0.0
        weight_map = weight_map + 1e-6  # avoid zero weights

        # Normalize to probability distribution
        weights_flat = weight_map.ravel()
        fluid_indices = np.where(fluid.ravel())[0]
        probs = weights_flat[fluid_indices]
        probs = probs / probs.sum()

        # 50% importance, 50% uniform
        n_imp = n // 2
        n_uni = n - n_imp
        imp_idx = rng.choice(fluid_indices, size=n_imp, replace=False, p=probs)
        uni_idx = rng.choice(fluid_indices, size=n_uni, replace=False)
        chosen_flat = np.concatenate([imp_idx, uni_idx])
        ji = chosen_flat // u.shape[1]
        ii = chosen_flat % u.shape[1]
    else:
        if idx.shape[0] < n:
            n = idx.shape[0]
        chosen = idx[rng.choice(idx.shape[0], size=n, replace=False)]
        ji = chosen[:, 0]
        ii = chosen[:, 1]

    if normalized_coords is None:
        raise ValueError("Pass normalized_coords from flatten_grid().")
    coords = normalized_coords[ji * u.shape[1] + ii]
    return {
        "coords": coords,
        "u": u[ji, ii],
        "v": v[ji, ii],
        "p": p[ji, ii],
    }


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------
def load_case_frame(cfg: CaseConfig, frame_name: str = "frame_18000.json") -> dict:
    path = os.path.join(cfg.case_dir, "frames", frame_name)
    return load_frame(path)


if __name__ == "__main__":
    # Smoke test: load the cylinder Re=100 frame and print diagnostics.
    from config import cylinder_re100
    cfg = cylinder_re100()
    print(f"Case: {cfg.name}  shape={cfg.shape_type}")
    print(f"Full grid: {cfg.NX} x {cfg.NY}  ds={cfg.ds}  "
          f"downsampled: {cfg.nx} x {cfg.ny}")
    print(f"Re={cfg.re} tau={cfg.tau} nu={cfg.nu:.5f} u_inf={cfg.u_inflow}")
    print(f"Geometry (ds-index): {cfg.geometry}")

    fr = load_case_frame(cfg)
    u, v, p, obst = fr["u"], fr["v"], fr["p"], fr["obstacle"]
    print(f"Frame fields: u[{u.shape}] v[{v.shape}] p[{p.shape}] "
          f"obstacle[{obst.shape}]")
    print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"  v range: [{v.min():.4f}, {v.max():.4f}]")
    print(f"  p range: [{p.min():.4f}, {p.max():.4f}]")
    print(f"  obstacle nodes: {int(obst.sum())}")

    Xn, Yn, _, _ = grid_coords(cfg)
    coords = flatten_grid(Xn, Yn)
    print(f"Full grid coords: {coords.shape}")

    col = make_collocation(cfg, 5000, seed=0, obstacle=obst)
    print(f"Collocation points: {col.shape}")

    bnd = make_boundary(cfg, 400, seed=1)
    print(f"Boundary points: {bnd.shape}")

    sens = subsample_sensors(u, v, p, obst, 1000, seed=2,
                             normalized_coords=coords)
    print(f"Sensor subsample: {sens['coords'].shape}  "
          f"u target range [{sens['u'].min():.4f}, {sens['u'].max():.4f}]")
    print("OK")
