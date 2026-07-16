"""Export the time-parametric PINN for the web (Phase 6.8).

Consumes output/cavity/pinn_temporal/model_temporal.pt and writes:

  * docs/assets/data/cavity/pinn_temporal_re{re}.bin (+ .gz)
        Per-Re temporal field sequence: (n_frames, 3, R, R) with channels
        (u, v, obstacle). Matches the LBM viewer's per-frame binary format so
        the same FlowViewer engine animates the surrogate over time.
  * docs/assets/data/cavity/pinn_grid.bin (+ .gz)  (shared normalized grid)
  * docs/assets/data/cavity/pinn_temporal_model.onnx  (live ONNX inference)

Run train_temporal.py first to produce the checkpoint.
"""

import os
import sys
import gzip
import struct

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_OUT = os.path.join(PROJECT_ROOT, "docs", "assets", "data", "cavity")
TARGET_RES = 96
MAGIC = 0x4C424D31
DTYPE_FLAG = 1
RE_VALUES = [100.0, 400.0, 1000.0]
RE_MIN, RE_MAX = 100.0, 1000.0


def write_binary(path, frames, nx, ny, n_channels):
    arr = np.asarray(frames, dtype=np.float32)
    arr16 = arr.astype(np.float16).view(np.uint16)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIIII", MAGIC, arr.shape[0], nx, ny,
                            n_channels, DTYPE_FLAG))
        f.write(arr16.tobytes(order="C"))
    raw = os.path.getsize(path)
    with open(path, "rb") as f:
        gz_path = path + ".gz"
        with gzip.open(gz_path, "wb", compresslevel=9) as gz:
            gz.write(f.read())
    gz_size = os.path.getsize(gz_path)
    return raw, gz_size


def _bilinear(field, target):
    from scipy.ndimage import zoom
    ny, nx = field.shape
    if ny == target and nx == target:
        return np.ascontiguousarray(field, dtype=np.float32)
    fx = target / nx
    fy = target / ny
    return np.ascontiguousarray(zoom(field.astype(np.float32), (fy, fx), order=1),
                                dtype=np.float32)


def export_temporal(time_res=51):
    import torch
    from data.loader import grid_coords, flatten_grid
    from models.pinn import ParametricPINN, predict
    from config import cavity_re100, normalize_re

    pt_path = os.path.join(PROJECT_ROOT, "output", "cavity", "pinn_temporal",
                           "model_temporal.pt")
    if not os.path.exists(pt_path):
        print(f"  SKIP temporal: {pt_path} not found (run train_temporal.py)")
        return

    ckpt = torch.load(pt_path, map_location="cpu")
    sigma_val = ckpt["sigma"]
    sigmas_arg = ckpt.get("sigmas", None)
    if sigmas_arg is not None:
        sigmas_arg = tuple(float(s) for s in sigmas_arg)
    elif isinstance(sigma_val, (list, tuple)):
        sigmas_arg = tuple(float(s) for s in sigma_val)
    model = ParametricPINN(
        n_params=2,
        hidden=int(ckpt["hidden"]),
        n_layers=int(ckpt["n_layers"]),
        n_freqs=int(ckpt["n_freqs"]),
        sigma=float(sigma_val) if sigmas_arg is None else 5.0,
        sigmas=sigmas_arg,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cfg = cavity_re100()
    Xn, Yn, _, _ = grid_coords(cfg)
    coords_all = flatten_grid(Xn, Yn)
    coords_t = torch.from_numpy(coords_all).float()
    device = torch.device("cpu")
    model.to(device)

    # Shared normalized grid (reuse TARGET_RES resolution).
    Xn_r = _bilinear(Xn, TARGET_RES)
    Yn_r = _bilinear(Yn, TARGET_RES)
    grid = np.stack([Xn_r, Yn_r], axis=0)[None, ...]
    gpath = os.path.join(DATA_OUT, "pinn_grid.bin")
    raw_g, gz_g = write_binary(gpath, grid, TARGET_RES, TARGET_RES, 2)
    print(f"  pinn_grid.bin  {grid.shape}  raw={raw_g/1e6:.2f}MB gz={gz_g/1e6:.2f}MB")

    for re in RE_VALUES:
        re_norm = normalize_re(re)
        frames = []
        with torch.no_grad():
            for fi in range(time_res):
                t_norm = fi / (time_res - 1) if time_res > 1 else 0.0
                aug = torch.cat([
                    coords_t,
                    torch.full((coords_t.shape[0], 1), re_norm),
                    torch.full((coords_t.shape[0], 1), t_norm),
                ], dim=1).to(device)
                u, v, p = predict(model, aug)
                u = _bilinear(u.cpu().numpy().reshape(cfg.ny, cfg.nx), TARGET_RES)
                v = _bilinear(v.cpu().numpy().reshape(cfg.ny, cfg.nx), TARGET_RES)
                # Obstacle is time-independent; reuse the normalized grid mask.
                obst = (np.abs(Xn_r) >= 1.0).astype(np.float32) * 0.0  # placeholder
                frames.append(np.stack([u, v, obst], axis=0))
        stack = np.stack(frames, axis=0)
        out_path = os.path.join(DATA_OUT, f"pinn_temporal_re{int(re)}.bin")
        raw, gz = write_binary(out_path, stack, TARGET_RES, TARGET_RES, 3)
        print(f"  pinn_temporal_re{int(re)}.bin  {stack.shape}  "
              f"raw={raw/1e6:.2f}MB gz={gz/1e6:.2f}MB")


def export_onnx():
    import torch
    from models.pinn import ParametricPINN

    pt_path = os.path.join(PROJECT_ROOT, "output", "cavity", "pinn", "temporal", "v1",
                           "model.pt")
    if not os.path.exists(pt_path):
        print(f"  SKIP temporal onnx: {pt_path} not found")
        return
    ckpt = torch.load(pt_path, map_location="cpu")
    sigma_val = ckpt["sigma"]
    sigmas_arg = None
    if isinstance(sigma_val, (list, tuple)):
        sigmas_arg = tuple(float(s) for s in sigma_val)
    model = ParametricPINN(
        n_params=2,
        hidden=int(ckpt["hidden"]),
        n_layers=int(ckpt["n_layers"]),
        n_freqs=int(ckpt["n_freqs"]),
        sigma=float(sigma_val) if sigmas_arg is None else 5.0,
        sigmas=sigmas_arg,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_path = os.path.join(DATA_OUT, "pinn_temporal_model.onnx")
    dummy = torch.randn(9216, 4)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
        opset_version=13, do_constant_folding=True,
    )
    size = os.path.getsize(out_path)
    print(f"  pinn_temporal_model.onnx  {size/1e6:.2f} MB")


def export_cavity_temporal():
    os.makedirs(DATA_OUT, exist_ok=True)
    print("Exporting temporal PINN field sequences...")
    export_temporal()
    print("Exporting temporal ONNX model...")
    export_onnx()


if __name__ == "__main__":
    export_cavity_temporal()
