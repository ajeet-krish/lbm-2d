"""Export LBM frame data for browser-side visualization.

Generates compact binary files consumed by docs/assets/js/flow-viewer.js:

  docs/assets/data/{case}/lbm_re{label}.bin   -- N frames, nx*ny, 3 channels (u, v, obstacle)

The velocity-magnitude canvas viewer only needs u, v (for the contour + streamlines)
and the obstacle mask (to keep streamlines from crossing solid walls). Pressure and
vorticity are excluded to keep file sizes small; they can be added back later if a
field selector is reintroduced.

Custom binary format (.bin):
  offset 0 : uint32 magic = 0x4C424D31
  offset 4 : uint32 n_frames
  offset 8 : uint32 nx
  offset 12: uint32 ny
  offset 16: uint32 n_channels
  offset 20: uint32 dtype_flag (0=float32, 1=float16 uint16)
  offset 24: float16 little-endian data, layout [frame][channel][y][x]

The data directory name (e.g. "cavity") matches the website case page, so the
viewer can be pointed at assets/data/{case}/ with no per-case code.

Usage:
  python3 export_web_data.py            # export LBM for all cases
  python3 export_web_data.py --pinn     # also export cavity PINN sweep + ONNX
"""

import os
import sys
import json
import struct
import gzip
import argparse
import numpy as np
from scipy.ndimage import zoom

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "docs", "assets", "data")
MAGIC = 0x4C424D31
DTYPE_FLAG = 1  # float16 (stored as little-endian uint16) keeps files ~half size


# Case -> output subdir + (output_subdir, web_label) configs.
# web_label becomes the filename suffix: lbm_{web_label}.bin
CASES = {
    "cavity": {
        "src": "cavity",
        "max_dim": 96,
        "configs": [("re100", "re100"), ("re400", "re400"), ("re1000", "re1000")],
    },
    "cylinder": {
        "src": "cylinder",
        "max_dim": 128,
        "configs": [("re100", "100"), ("re200", "200"), ("re1000", "1000")],
    },
    "step": {
        "src": "step",
        "max_dim": 128,
        "configs": [("re100", "100"), ("re200", "200"), ("re400", "400")],
    },
    "flatplate": {
        "src": "flatplate",
        "max_dim": 128,
        "configs": [
            ("re1000_aoa0", "aoa0"),
            ("re1000_aoa5", "aoa5"),
            ("re1000_aoa10", "aoa10"),
            ("re500_aoa0", "re500"),
            ("re2000_aoa0", "re2000"),
        ],
    },
    "orifice_plate": {
        "src": "orifice_plate",
        "max_dim": 128,
        "configs": [
            ("re100_1p1h", "1p1h"),
            ("re100_1p3h", "1p3h"),
            ("re100_2p", "2p"),
            ("re100_3p", "3p"),
        ],
    },
    "periodic_hills": {
        "src": "periodic_hills",
        "max_dim": 128,
        "configs": [("re100", "100"), ("re1000", "1000"), ("re2800", "2800")],
    },
    "square_cylinder": {
        "src": "square_cylinder",
        "max_dim": 128,
        "configs": [("re200", "200")],
    },
    "side_by_side": {
        "src": "side_by_side",
        "max_dim": 128,
        "configs": [("re100_sd20", "sd20"), ("re100_sd30", "sd30"), ("re100_sd50", "sd50")],
    },
    "cylinder_near_wall": {
        "src": "cylinder_near_wall",
        "max_dim": 128,
        "configs": [
            ("re100_gap10", "gap10"),
            ("re100_gap20", "gap20"),
            ("re100_gap40", "gap40"),
        ],
    },
    "rotating_cylinder": {
        "src": "rotating_cylinder",
        "max_dim": 128,
        "configs": [("re100_w5", "w5"), ("re100_w10", "w10"), ("re100_w20", "w20")],
    },
    "urban": {
        "src": "urban",
        "max_dim": 128,
        "configs": [
            ("side_ar0.3_re100", "side_a03"),
            ("side_ar0.5_re100", "side_a05"),
            ("side_ar0.8_re100", "side_a08"),
            ("topdown_re100", "topdown"),
            ("downwash_re100", "downwash"),
        ],
    },
}


def bilinear_resize(field, target_nx, target_ny):
    ny, nx = field.shape
    if ny == target_ny and nx == target_nx:
        return np.ascontiguousarray(field, dtype=np.float32)
    fx = target_nx / nx
    fy = target_ny / ny
    out = zoom(field.astype(np.float32), (fy, fx), order=1)
    return np.ascontiguousarray(out, dtype=np.float32)


def compute_target(nx, ny, max_dim):
    if max(nx, ny) <= max_dim:
        return nx, ny
    scale = max_dim / max(nx, ny)
    return max(1, int(round(nx * scale))), max(1, int(round(ny * scale)))


def write_binary(path, frames, nx, ny, n_channels):
    """frames: (n_frames, n_channels, ny, nx) float32; written as float16."""
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


def _frame_files(frame_dir):
    files = [
        f for f in os.listdir(frame_dir)
        if f.startswith("frame_") and f.endswith(".json")
    ]

    def step_of(fn):
        s = fn[len("frame_"):]
        s = s[: s.index(".")]
        return int(s)

    files.sort(key=step_of)
    return files


def export_lbm_case(case_name, cfg):
    src = cfg["src"]
    out_dir = os.path.join(DATA_ROOT, case_name)
    os.makedirs(out_dir, exist_ok=True)
    for subdir, label in cfg["configs"]:
        frame_dir = os.path.join(PROJECT_ROOT, "output", src, subdir, "frames")
        if not os.path.isdir(frame_dir):
            print(f"  SKIP {case_name}/{label}: {frame_dir} not found")
            continue
        files = _frame_files(frame_dir)
        if not files:
            print(f"  SKIP {case_name}/{label}: no frames")
            continue

        # Load frames, skipping any that fail to parse (diverged runs may emit
        # NaN/Infinity or truncated JSON that json.load rejects) and any
        # all-zero velocity frames that are not the initial rest state (a source
        # data bug in some cases, e.g. periodic hills wrote zeroed frames mid-run).
        loaded = []
        for fn in files:
            try:
                with open(os.path.join(frame_dir, fn)) as fh:
                    d = json.load(fh)
            except (json.JSONDecodeError, ValueError):
                continue
            u = d.get("u")
            v = d.get("v")
            if u is not None and v is not None and len(loaded) > 0:
                arr_u = np.asarray(u, dtype=np.float32)
                arr_v = np.asarray(v, dtype=np.float32)
                if np.max(np.abs(arr_u)) < 1e-12 and np.max(np.abs(arr_v)) < 1e-12:
                    continue  # corrupted zero-velocity frame; drop it
            loaded.append((fn, d))

        if not loaded:
            print(f"  SKIP {case_name}/{label}: no parseable frames")
            continue

        first = loaded[0][1]
        nx0, ny0 = first["nx"], first["ny"]
        tnx, tny = compute_target(nx0, ny0, cfg["max_dim"])

        ch_u, ch_v, ch_obs = [], [], []
        for fn, d in loaded:
            fld = lambda key: np.array(d[key], dtype=np.float32).reshape(d["ny"], d["nx"])
            ch_u.append(bilinear_resize(fld("u"), tnx, tny))
            ch_v.append(bilinear_resize(fld("v"), tnx, tny))
            ch_obs.append(bilinear_resize(fld("obstacle"), tnx, tny))

        obs_stack = np.stack(ch_obs, 0)
        combined = np.stack(
            [np.stack(ch_u, 0), np.stack(ch_v, 0), obs_stack], axis=1
        )  # (n_frames, 3, ny, nx)
        out_path = os.path.join(out_dir, f"lbm_{label}.bin")
        raw, gz = write_binary(out_path, combined, tnx, tny, 3)
        skipped = len(files) - len(loaded)
        note = f" ({skipped} skipped)" if skipped else ""
        print(f"  {case_name}/lbm_{label}.bin  {combined.shape}  "
              f"raw={raw/1e6:.2f}MB gz={gz/1e6:.2f}MB ({len(loaded)} frames{note})")


# --- PINN exports (cavity only; kept for the trained surrogate) ------------

def export_pinn(re):
    from cases.cavity.export_sweep import export_pinn as _ep  # noqa
    _ep(re)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", action="store_true", help="also export cavity PINN sweep + ONNX")
    ap.add_argument("--case", default=None, help="export a single case only")
    args = ap.parse_args()

    os.makedirs(DATA_ROOT, exist_ok=True)
    print("Exporting LBM frames for all cases...")
    for case_name, cfg in CASES.items():
        if args.case and case_name != args.case:
            continue
        print(f"Case: {case_name}")
        export_lbm_case(case_name, cfg)

    if args.pinn:
        # Reuse the cavity PINN export logic via the helper module.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from cases.cavity.export_sweep import export_cavity_pinn
        export_cavity_pinn()

    print("Done.")


if __name__ == "__main__":
    main()
