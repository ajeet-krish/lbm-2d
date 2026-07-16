"""Configuration and case metadata for the PINN surrogate suite.

This module is torch-free (numpy only) so it can be imported in environments
without PyTorch installed. It reads the C++ solver's meta.json + frame_*.json
and derives everything the PINN needs: the downsampled grid dimensions, the
downsample factor, and the obstacle geometry in downsampled index space.

Coordinate convention
---------------------
The C++ solver writes downsampled frames where downsampled node (i, j) maps to
the full-grid node (x = i*ds, y = j*ds). We expose PHYSICAL coordinates
(x_phys = i*ds, y_phys = j*ds) and a normalization that maps the physical domain
[0, NX] x [0, NY] to [-1, 1] x [-1, 1]. All PINN inputs are normalized.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Case directories
CYLINDER_RE100_DIR = os.path.join(OUTPUT_DIR, "cylinder", "re100")
CAVITY_RE100_DIR = os.path.join(OUTPUT_DIR, "cavity", "re100")
CAVITY_RE400_DIR = os.path.join(OUTPUT_DIR, "cavity", "re400")
CAVITY_RE1000_DIR = os.path.join(OUTPUT_DIR, "cavity", "re1000")


@dataclass
class CaseConfig:
    """Resolved parameters for one simulation case."""

    name: str
    shape_type: str
    # Full-grid dimensions (from meta.json).
    NX: int
    NY: int
    re: float
    tau: float
    u_inflow: float
    length_scale: float
    # Downsample factor ds = max(1, NX / 100) used by the C++ frame writer.
    ds: int
    # Downsampled frame dimensions.
    nx: int
    ny: int
    # Obstacle geometry in DOWNSAMPLED index space (i, j), per shape_type.
    geometry: dict = field(default_factory=dict)
    # Optional directory containing frames/ and meta.json.
    case_dir: Optional[str] = None

    @property
    def nu(self) -> float:
        """Kinematic viscosity in lattice units: nu = (tau - 0.5) / 3."""
        return (self.tau - 0.5) / 3.0

    def physical_coords(self, i: float, j: float):
        """Map downsampled index (i, j) -> full-grid physical (x, y)."""
        return i * self.ds, j * self.ds

    def normalize(self, X: np.ndarray, Y: np.ndarray):
        """Map physical coords (any shape) to [-1, 1] domain-normalized space."""
        Xn = X / self.NX * 2.0 - 1.0
        Yn = Y / self.NY * 2.0 - 1.0
        return Xn, Yn

    def denormalize(self, Xn: np.ndarray, Yn: np.ndarray):
        """Inverse of normalize()."""
        X = (Xn + 1.0) / 2.0 * self.NX
        Y = (Yn + 1.0) / 2.0 * self.NY
        return X, Y


def downsample_factor(NX: int) -> int:
    """Replicate the C++ frame writer: ds = max(1, NX / 100)."""
    return max(1, NX // 100)


def _geometry_for(shape_type: str, NX: int, NY: int, ds: int) -> dict:
    """Obstacle geometry in downsampled index space, matching src/main.cpp."""
    if shape_type == "cylinder":
        cx = (NX / 4.0) / ds
        cy = (NY / 2.0 + 1.0) / ds
        radius = (NY / 10.0) / ds
        return {"cx": cx, "cy": cy, "radius": radius}
    if shape_type == "lid-driven-cavity":
        # No interior obstacle -- flow enclosed by walls.
        # Lid occupies the top wall row (y = NY-1).
        return {"shape_type": "lid-driven-cavity"}
    if shape_type == "backward-facing-step":
        # Step at x = NX/4, y = 0 to h_step (NY/3).
        h_step = NY // 3
        return {"shape_type": "backward-facing-step", "h_step": h_step}
    if shape_type == "orifice-plate":
        # Orifice plate at x = NX/2, hole centered at y = NY/2.
        hole_w = NX // 8  # default, overridden per-config
        return {"shape_type": "orifice-plate", "hole_w": hole_w}
    return {}


def from_meta(meta_path: str, case_dir: Optional[str] = None,
              frame_path: Optional[str] = None) -> CaseConfig:
    """Build a CaseConfig from meta.json (+ optional frame for nx/ny)."""
    with open(meta_path) as f:
        meta = json.load(f)
    NX = int(meta["nx"])
    NY = int(meta["ny"])
    ds = downsample_factor(NX)
    shape_type = meta.get("shape_type", "cylinder")

    # Downsampled dims: prefer an actual frame, else compute from ds.
    if frame_path is not None and os.path.exists(frame_path):
        with open(frame_path) as f:
            fr = json.load(f)
        nx = int(fr["nx"])
        ny = int(fr["ny"])
    else:
        nx = (NX + ds - 1) // ds
        ny = (NY + ds - 1) // ds

    geom = _geometry_for(shape_type, NX, NY, ds)
    name = os.path.basename(os.path.dirname(meta_path)) or shape_type
    return CaseConfig(
        name=name,
        shape_type=shape_type,
        NX=NX, NY=NY,
        re=float(meta.get("re", 100.0)),
        tau=float(meta.get("tau", 0.68)),
        u_inflow=float(meta.get("u_inflow", 0.1)),
        length_scale=float(meta.get("length_scale", 60.0)),
        ds=ds, nx=nx, ny=ny,
        geometry=geom,
        case_dir=case_dir,
    )


def cylinder_re100() -> CaseConfig:
    """Convenience constructor for Cylinder Re=100."""
    meta = os.path.join(CYLINDER_RE100_DIR, "meta.json")
    frame = os.path.join(CYLINDER_RE100_DIR, "frames", "frame_18000.json")
    return from_meta(meta, case_dir=CYLINDER_RE100_DIR, frame_path=frame)


def cavity_re100() -> CaseConfig:
    """Convenience constructor for Cavity Re=100."""
    meta = os.path.join(CAVITY_RE100_DIR, "meta.json")
    frame = os.path.join(CAVITY_RE100_DIR, "frames", "frame_12800.json")
    return from_meta(meta, case_dir=CAVITY_RE100_DIR, frame_path=frame)


def cavity_re400() -> CaseConfig:
    """Convenience constructor for Cavity Re=400."""
    meta = os.path.join(CAVITY_RE400_DIR, "meta.json")
    frame = os.path.join(CAVITY_RE400_DIR, "frames", "frame_12800.json")
    return from_meta(meta, case_dir=CAVITY_RE400_DIR, frame_path=frame)


def cavity_re1000() -> CaseConfig:
    """Convenience constructor for Cavity Re=1000."""
    meta = os.path.join(CAVITY_RE1000_DIR, "meta.json")
    frame = os.path.join(CAVITY_RE1000_DIR, "frames", "frame_12800.json")
    return from_meta(meta, case_dir=CAVITY_RE1000_DIR, frame_path=frame)


def all_cavity_configs():
    """Return list of cavity configs for multi-Re training."""
    return [cavity_re100(), cavity_re400()]


# Reynolds-number normalization range used across the parametric/temporal PINN
# surrogates: Re is mapped to [0, 1] over [RE_MIN, RE_MAX].
RE_MIN = 100.0
RE_MAX = 1000.0


def normalize_re(re: float) -> float:
    """Map a Reynolds number to [0, 1] over the training range [100, 1000]."""
    return (re - RE_MIN) / (RE_MAX - RE_MIN)
