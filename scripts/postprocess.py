#!/usr/bin/env python3
"""
LBM-2D Post-Processor
Converts VTK simulation output to JSON for web viewer and PNG frames.

Usage:
    python3 postprocess.py output/re100              # PNG frames
    python3 postprocess.py output/re100 --json       # JSON for web viewer
    python3 postprocess.py output/re100 --json --every 5  # subsample
"""
import struct, sys, os, json, re, argparse
from pathlib import Path
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def read_vtk_ascii_structured(path):
    """Read ASCII VTK structured points file, return velocity and density grids."""
    with open(path) as f:
        text = f.read()

    # Parse DIMENSIONS
    m = re.search(r'DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)', text)
    if not m:
        raise ValueError("Could not find DIMENSIONS")
    nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
    npts = nx * ny * nz

    # Find SCALARS VelocityMagnitude section
    blocks = text.split('SCALARS')
    vel_data = None
    dens_data = None

    for block in blocks:
        if 'VelocityMagnitude' in block:
            lines = block.strip().split('\n')
            # Skip header lines (SCALARS + LOOKUP_TABLE)
            data_lines = []
            for line in lines[1:]:
                if 'LOOKUP_TABLE' in line:
                    continue
                data_lines.append(line)
            raw = ' '.join(data_lines).strip()
            vel_data = np.array([float(v) for v in raw.split()])
            if len(vel_data) > npts:
                vel_data = vel_data[:npts]

        if 'Density' in block and 'VelocityMagnitude' not in block:
            lines = block.strip().split('\n')
            data_lines = []
            for line in lines[1:]:
                if 'LOOKUP_TABLE' in line:
                    continue
                data_lines.append(line)
            raw = ' '.join(data_lines).strip()
            dens_data = np.array([float(v) for v in raw.split()])
            if len(dens_data) > npts:
                dens_data = dens_data[:npts]

    # Also try VECTORS
    m_vec = re.search(r'VECTORS Velocity double\n([\s\S]+?)(?:SCALARS|$)', text)
    if m_vec:
        raw_vec = m_vec.group(1).strip()
        parts = raw_vec.split()
        vec = np.array([float(v) for v in parts])
        if len(vec) >= npts * 3:
            u = vec[0::3].reshape((ny, nx))
            v = vec[1::3].reshape((ny, nx))
        else:
            u = np.zeros((ny, nx))
            v = np.zeros((ny, nx))
    else:
        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))

    if vel_data is not None:
        vel = vel_data.reshape((ny, nx))
    else:
        vel = np.sqrt(u**2 + v**2)

    if dens_data is not None:
        rho = dens_data.reshape((ny, nx))
    else:
        rho = np.ones((ny, nx))

    return {'nx': nx, 'ny': ny, 'velocity': vel.tolist(), 'density': rho.tolist(),
            'u': u.tolist(), 'v': v.tolist()}


def save_png(data, output_dir, frame):
    """Render velocity field as a PNG frame."""
    if not HAS_MPL:
        print("matplotlib not installed, skipping PNG export")
        return

    vel = np.array(data['velocity'])
    u = np.array(data['u'])
    v = np.array(data['v'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#0d1117')

    # Velocity magnitude with jet colormap
    im1 = ax1.imshow(vel, origin='lower', cmap='jet', aspect='auto',
                     vmin=0, vmax=max(vel.max(), 0.01))
    ax1.set_title('Velocity Magnitude', color='#c9d1d9', fontsize=11)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Streamlines
    y, x = np.mgrid[0:data['ny'], 0:data['nx']]
    step = max(1, data['nx'] // 50)
    ax2.streamplot(x[::step, ::step], y[::step, ::step],
                    u[::step, ::step], v[::step, ::step],
                    color=np.sqrt(u[::step, ::step]**2 + v[::step, ::step]**2),
                    cmap='jet', density=1.0, linewidth=0.8, arrowsize=0.8)
    ax2.set_title('Streamlines', color='#c9d1d9', fontsize=11)
    ax2.axis('off')
    ax2.set_facecolor('#0a0e14')

    plt.tight_layout()
    path = os.path.join(output_dir, f'frame_{frame:04d}.png')
    plt.savefig(path, dpi=120, facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"  Saved {path}")


def export_json(vtk_dir, output_path, every=1):
    """Convert VTK frames to a single JSON file for web viewer."""
    frames = []
    vtk_files = sorted(Path(vtk_dir).glob('frame_*.vtk'))
    print(f"Processing {len(vtk_files)} VTK files from {vtk_dir}")

    for i, vtk_path in enumerate(vtk_files):
        if i % every != 0:
            continue
        frame_match = re.search(r'frame_(\d+)', vtk_path.name)
        frame_num = int(frame_match.group(1)) if frame_match else i
        data = read_vtk_ascii_structured(str(vtk_path))

        # Downsample for web (every 4th point for 400x150 -> 100x38)
        vel = np.array(data['velocity'])
        downsample = max(1, data['nx'] // 100)
        vel_ds = vel[::downsample, ::downsample]

        frames.append({
            'frame': frame_num,
            'nx': vel_ds.shape[1],
            'ny': vel_ds.shape[0],
            'velocity': vel_ds.tolist(),
        })

    result = {'frames': frames, 'meta': {'nx': frames[0]['nx'], 'ny': frames[0]['ny']}}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f)
    print(f"Exported {len(frames)} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LBM-2D Post-Processor')
    parser.add_argument('input_dir', help='Directory containing VTK frames')
    parser.add_argument('--json', action='store_true', help='Export JSON for web viewer')
    parser.add_argument('--every', type=int, default=1, help='Subsample every N frames')
    parser.add_argument('--output', help='Output path for JSON (default: auto)')
    args = parser.parse_args()

    if args.json:
        re_match = re.search(r're(\d+)', args.input_dir)
        re_str = re_match.group(1) if re_match else 'sim'
        out_path = args.output or f'docs/assets/data/sim_re{re_str}.json'
        export_json(args.input_dir, out_path, every=args.every)
    else:
        vtk_files = sorted(Path(args.input_dir).glob('frame_*.vtk'))
        print(f"Found {len(vtk_files)} VTK files")
        for i, vtk_path in enumerate(vtk_files):
            if i % args.every != 0:
                continue
            data = read_vtk_ascii_structured(str(vtk_path))
            frame_match = re.search(r'frame_(\d+)', vtk_path.name)
            frame_num = frame_match.group(1) if frame_match else str(i)
            save_png(data, args.input_dir, frame_num)


if __name__ == '__main__':
    main()
