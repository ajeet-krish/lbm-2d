"""Data loading for the time-parametric PINN (Phase 6.8).

Loads the LBM frame time-series and produces 4-D input samples:

    (x_norm, y_norm, Re_n, t_n) -> (u, v, p)

where x_norm, y_norm are normalized to [-1, 1] via CaseConfig.normalize(),
Re_n = (Re - RE_MIN) / (RE_MAX - RE_MIN) in [0, 1], and t_n = frame_index /
(n_frames - 1) in [0, 1].

The training data is the full 51-frame LBM evolution at Re=100 and Re=400
(cavity first target), with importance sampling near high-gradient regions.
"""

import os

import numpy as np
from scipy.ndimage import gaussian_filter

from data.loader import load_frame, grid_coords, flatten_grid
from config import CaseConfig

# Re normalization range (matches train_cavity.py).
RE_MIN = 100.0
RE_MAX = 1000.0


def normalize_re(re: float) -> float:
    """Normalize Re to [0, 1]."""
    return (re - RE_MIN) / (RE_MAX - RE_MIN)


def list_frames(frame_dir: str):
    """Return frame_*.json paths sorted by integer step number."""
    files = [f for f in os.listdir(frame_dir)
             if f.startswith("frame_") and f.endswith(".json")]
    files.sort(key=lambda s: int(s.split("_")[1].split(".")[0]))
    return [os.path.join(frame_dir, f) for f in files]


def load_temporal_frames(frame_dir: str, n_frames_target=None):
    """Load all frames from a directory in step order.

    Returns a list of dicts with keys 'u', 'v', 'p', 'obstacle' (ny, nx).
    If n_frames_target is set, the sequence is subsampled evenly.
    """
    paths = list_frames(frame_dir)
    if n_frames_target and len(paths) > n_frames_target:
        idx = np.linspace(0, len(paths) - 1, n_frames_target).astype(int)
        paths = [paths[i] for i in idx]
    frames = []
    for p in paths:
        fr = load_frame(p)
        rec = {
            "u": fr["u"],
            "v": fr["v"],
            "p": fr.get("p", fr.get("rho")),
            "obstacle": fr["obstacle"],
        }
        frames.append(rec)
    return frames


def _importance_sample_4d(u, v, p, obstacle, coords_all, re_norm, t_norm,
                          n, rng):
    """Importance-sample sensor points in one frame, returning (coords_4d, uvp)."""
    fluid = obstacle < 0.5
    vmag = np.sqrt(u**2 + v**2)
    vmag_s = gaussian_filter(vmag, sigma=2.0)
    gx = np.gradient(vmag_s, axis=1)
    gy = np.gradient(vmag_s, axis=0)
    grad = np.sqrt(gx**2 + gy**2)
    wmap = grad.copy()
    wmap[obstacle > 0.5] = 0.0
    wmap = wmap + 1e-6

    fidx = np.where(fluid.ravel())[0]
    probs = wmap.ravel()[fidx]
    probs = probs / probs.sum()

    n_imp = n // 2
    n_uni = n - n_imp
    imp = rng.choice(fidx, size=n_imp, replace=False, p=probs)
    uni = rng.choice(fidx, size=n_uni, replace=False)
    chosen = np.concatenate([imp, uni])

    ji = chosen // u.shape[1]
    ii = chosen % u.shape[1]
    c2 = coords_all[ji * u.shape[1] + ii]  # (n, 2)
    c4 = np.concatenate([c2,
                         np.full((n, 1), re_norm),
                         np.full((n, 1), t_norm)], axis=1)
    uvp = np.stack([u[ji, ii], v[ji, ii], p[ji, ii]], axis=1)
    return c4, uvp


def build_temporal_sensors(cfg: CaseConfig, frame_dirs_by_re: dict,
                            re_values, n_per_frame=600, seed=0):
    """Build (coords_4d, uvp) sensor arrays across Re and time.

    coords_4d: (N, 4) = (x_norm, y_norm, re_norm, t_norm)
    uvp:       (N, 3) = (u, v, p)
    """
    Xn, Yn, _, _ = grid_coords(cfg)
    coords_all = flatten_grid(Xn, Yn)
    rng = np.random.default_rng(seed)

    all_coords, all_uvp = [], []
    for re in re_values:
        re_norm = normalize_re(re)
        frames = load_temporal_frames(frame_dirs_by_re[re])
        n_frames = len(frames)
        for fi, fr in enumerate(frames):
            t_norm = fi / (n_frames - 1) if n_frames > 1 else 0.0
            c4, uvp = _importance_sample_4d(
                fr["u"], fr["v"], fr["p"], fr["obstacle"],
                coords_all, re_norm, t_norm, n_per_frame, rng)
            all_coords.append(c4)
            all_uvp.append(uvp)
    return np.concatenate(all_coords, 0), np.concatenate(all_uvp, 0)


def build_temporal_collocation(cfg: CaseConfig, re_values, n, seed=1,
                               t_min=0.0, t_max=1.0):
    """Random collocation points for the unsteady PDE residual.

    Returns (N, 4) = (x_norm, y_norm, re_norm, t_norm). re_norm is sampled from
    the discrete set of normalized Re values (matching training data); t_norm is
    sampled uniformly in [t_min, t_max].
    """
    rng = np.random.default_rng(seed)
    re_norm_vals = [normalize_re(re) for re in re_values]
    xs = rng.uniform(-1.0, 1.0, n)
    ys = rng.uniform(-1.0, 1.0, n)
    rn = rng.choice(re_norm_vals, size=n)
    tn = rng.uniform(t_min, t_max, n)
    return np.stack([xs, ys, rn, tn], axis=1)
