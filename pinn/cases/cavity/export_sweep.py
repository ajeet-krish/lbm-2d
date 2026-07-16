"""Cavity PINN export helpers (sweep + ONNX) for export_web_data.py --pinn.

Kept separate so the main export script can focus on LBM frame data for all
cases while the cavity surrogate export remains available behind a flag.
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


def write_binary(path, frames, nx, ny, n_channels):
    arr = np.asarray(frames, dtype=np.float32)
    arr16 = arr.astype(np.float16).view(np.uint16)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIIII", MAGIC, arr.shape[0], nx, ny, n_channels, DTYPE_FLAG))
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
    return np.ascontiguousarray(zoom(field.astype(np.float32), (fy, fx), order=1), dtype=np.float32)


def export_pinn(re):
    src = os.path.join(PROJECT_ROOT, "output", "cavity", "pinn", "stable", "v3",
                       f"prediction_re{re}.npz")
    if not os.path.exists(src):
        print(f"  SKIP pinn_re{re}: {src} not found")
        return
    data = np.load(src)
    u = _bilinear(data["u_pred"], TARGET_RES)
    v = _bilinear(data["v_pred"], TARGET_RES)
    p = _bilinear(data["p_pred"], TARGET_RES)
    combined = np.stack([u, v, p], axis=0)[None, ...]
    out_path = os.path.join(DATA_OUT, f"pinn_re{re}.bin")
    raw, gz = write_binary(out_path, combined, TARGET_RES, TARGET_RES, 3)
    print(f"  pinn_re{re}.bin  {combined.shape}  raw={raw/1e6:.2f}MB gz={gz/1e6:.2f}MB")


def export_sweep(re_list, step=15):
    import torch
    from data.loader import grid_coords, flatten_grid
    from models.pinn import ParametricPINN, predict
    from config import cavity_re100

    pt_path = os.path.join(PROJECT_ROOT, "output", "cavity", "pinn",
                           "stable", "v3", "model.pt")
    if not os.path.exists(pt_path):
        print(f"  SKIP sweep: {pt_path} not found")
        return
    ckpt = torch.load(pt_path, map_location="cpu")
    model = ParametricPINN(
        n_params=1,
        hidden=int(ckpt["hidden"]),
        n_layers=int(ckpt["n_layers"]),
        n_freqs=int(ckpt["n_freqs"]),
        sigma=float(ckpt["sigma"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cfg = cavity_re100()
    Xn, Yn, _, _ = grid_coords(cfg)
    coords_all = flatten_grid(Xn, Yn)
    coords_t = torch.from_numpy(coords_all).float()

    RE_MIN, RE_MAX = 100.0, 1000.0
    frames = []
    device = torch.device("cpu")
    model.to(device)
    with torch.no_grad():
        for re in re_list:
            re_norm = (re - RE_MIN) / (RE_MAX - RE_MIN)
            aug = torch.cat([coords_t,
                             torch.full((coords_t.shape[0], 1), re_norm)], dim=1).to(device)
            u, v, p = predict(model, aug)
            u = _bilinear(u.cpu().numpy().reshape(cfg.ny, cfg.nx), TARGET_RES)
            v = _bilinear(v.cpu().numpy().reshape(cfg.ny, cfg.nx), TARGET_RES)
            p = _bilinear(p.cpu().numpy().reshape(cfg.ny, cfg.nx), TARGET_RES)
            frames.append(np.stack([u, v, p], axis=0))
    stack = np.stack(frames, axis=0)
    out_path = os.path.join(DATA_OUT, "pinn_sweep.bin")
    raw, gz = write_binary(out_path, stack, TARGET_RES, TARGET_RES, 3)
    print(f"  pinn_sweep.bin  {stack.shape}  raw={raw/1e6:.2f}MB gz={gz/1e6:.2f}MB")

    Xn_r = _bilinear(Xn, TARGET_RES)
    Yn_r = _bilinear(Yn, TARGET_RES)
    grid = np.stack([Xn_r, Yn_r], axis=0)[None, ...]
    grid_path = os.path.join(DATA_OUT, "pinn_grid.bin")
    raw_g, gz_g = write_binary(grid_path, grid, TARGET_RES, TARGET_RES, 2)
    print(f"  pinn_grid.bin  {grid.shape}  raw={raw_g/1e6:.2f}MB gz={gz_g/1e6:.2f}MB")


def export_onnx():
    import torch
    from models.pinn import ParametricPINN

    pt_path = os.path.join(PROJECT_ROOT, "output", "cavity", "pinn", "stable", "v3",
                           "model.pt")
    if not os.path.exists(pt_path):
        print(f"  SKIP onnx: {pt_path} not found")
        return
    ckpt = torch.load(pt_path, map_location="cpu")
    model = ParametricPINN(
        n_params=1,
        hidden=int(ckpt["hidden"]),
        n_layers=int(ckpt["n_layers"]),
        n_freqs=int(ckpt["n_freqs"]),
        sigma=float(ckpt["sigma"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_path = os.path.join(DATA_OUT, "pinn_model.onnx")
    dummy = torch.randn(9216, 3)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
        opset_version=13,
        do_constant_folding=True,
    )
    size = os.path.getsize(out_path)
    print(f"  pinn_model.onnx  {size/1e6:.2f} MB")


def export_cavity_pinn():
    os.makedirs(DATA_OUT, exist_ok=True)
    print("Exporting PINN steady-state...")
    export_pinn(100)
    export_pinn(400)
    print("Exporting PINN Re-sweep...")
    re_list = list(range(100, 401, 15))
    export_sweep(re_list)
    print("Exporting ONNX model...")
    export_onnx()
