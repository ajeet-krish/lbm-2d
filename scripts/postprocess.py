#!/usr/bin/env python3
"""
LBM-2D Post-Processor
Reads JSON frame output and produces:
  - PNG renders (contour, streamlines, or split)
  - Strouhal number from force history (Welch FFT)
  - Obstacle overlay (strict black)
  - Pressure contour option

Usage:
    python3 scripts/postprocess.py output/re100
    python3 scripts/postprocess.py output/step_re100 --split --cmap coolwarm
    python3 scripts/postprocess.py output/re100 --last-only
    python3 scripts/postprocess.py output/re100 --strouhal
    python3 scripts/postprocess.py output/re100 --split --cmap jet --strouhal
    python3 scripts/postprocess.py output/ribs_re100 --field pressure
"""

import os, sys, json, re, argparse, subprocess
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Per-case colormap configuration
# ---------------------------------------------------------------------------
CASE_CMAPS = {
    'cylinder':              ('jet', 'jet'),
    'cavity':                ('viridis', 'viridis'),
    'backward-facing-step':  ('coolwarm', 'coolwarm'),
    'ribbed-channel':        ('plasma', 'plasma'),
    'urban-canyon':          ('viridis', 'viridis'),
    'urban-side':            ('viridis', 'viridis'),
    'urban-topdown':         ('viridis', 'viridis'),
    'building-downwash':     ('RdBu', 'RdBu'),
    'flat-plate':            ('jet', 'jet'),
    'square-cylinder':       ('jet', 'jet'),
    'periodic-hills':        ('viridis', 'viridis'),
    'cylinder-near-wall':    ('jet', 'jet'),
    'side-by-side':          ('jet', 'jet'),
    'rotating-cylinder':     ('jet', 'jet'),
    'orifice-plate':         ('coolwarm', 'coolwarm'),
    'sports-ball':           ('jet', 'jet'),
}

# Detect shape type from meta.json or directory name
def _detect_shape(meta, output_dir=None):
    st = meta.get('shape_type', '')
    if st:
        return st if st else 'cylinder'
    # Fallback: infer from directory name
    if output_dir:
        dname = os.path.basename(output_dir)
        pname = os.path.basename(os.path.dirname(output_dir))
        # Check parent directory for case type
        if pname == 'cylinder':
            return 'cylinder'
        if pname == 'step':
            return 'backward-facing-step'
        if pname == 'ribs':
            return 'ribbed-channel'
        if pname == 'ahmed_body':
            return 'ahmed-body'
        if pname == 'downwash':
            return 'building-downwash'
        if pname == 'cavity':
            return 'cavity'
        if pname == 'flatplate':
            return 'flat-plate'
        if pname == 'square_cylinder':
            return 'square-cylinder'
        if pname == 'periodic_hills':
            return 'periodic-hills'
        if pname == 'cylinder_near_wall':
            return 'cylinder-near-wall'
        if pname == 'side_by_side':
            return 'side-by-side'
        if pname == 'rotating_cylinder':
            return 'rotating-cylinder'
        if pname == 'orifice_plate':
            return 'orifice-plate'
        if pname == 'sports_ball':
            return 'sports-ball'
        if pname == 'urban':
            if 'side' in dname:
                return 'urban-side'
            if 'downwash' in dname:
                return 'building-downwash'
            return 'urban-canyon'
        # Legacy flat directory names
        if 'urban_side' in dname:
            return 'urban-side'
        if 'urban_topdown' in dname:
            return 'urban-topdown'
        if 'cylinder' not in dname and 'step' in dname:
            return 'backward-facing-step'
        if 'ribs' in dname:
            return 'ribbed-channel'
        if 'urban' in dname:
            return 'urban-canyon'
        if 'downwash' in dname:
            return 'building-downwash'
        if 'flatplate' in dname:
            return 'flat-plate'
        if 'square_cylinder' in dname:
            return 'square-cylinder'
        if 'periodic_hills' in dname:
            return 'periodic-hills'
        if 'cylinder_near_wall' in dname:
            return 'cylinder-near-wall'
        if 'side_by_side' in dname:
            return 'side-by-side'
        if 'rotating_cylinder' in dname:
            return 'rotating-cylinder'
        if 'cavity' in dname:
            return 'cavity'
    return 'cylinder'
DEFAULT_CMAP = 'jet'


def _load_meta(output_dir):
    path = os.path.join(output_dir, 'meta.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _list_frames(output_dir):
    frames_dir = os.path.join(output_dir, 'frames')
    if not os.path.isdir(frames_dir):
        return []
    files = list(Path(frames_dir).glob('frame_*.json'))
    def _num_key(p):
        m = re.search(r'frame_(\d+)', p.name)
        return int(m.group(1)) if m else 0
    files.sort(key=_num_key)
    return files


def _load_frame(path):
    with open(path) as f:
        raw = json.load(f)
    raw['u'] = np.array(raw.get('u', raw.get('velocity', [])))
    raw['v'] = np.array(raw.get('v', []))
    raw['velocity'] = np.array(raw.get('velocity', []))
    raw['rho'] = np.array(raw.get('rho', []))
    raw['obstacle'] = np.array(raw.get('obstacle', []), dtype=bool)
    raw['omega'] = np.array(raw.get('omega', []))
    raw['nx'] = int(raw.get('nx', 0))
    raw['ny'] = int(raw.get('ny', 0))
    return raw


def _resolve_cmap(cmap_arg, meta, output_dir='', field='velocity'):
    if cmap_arg:
        return cmap_arg
    shape = _detect_shape(meta, output_dir)
    pair = CASE_CMAPS.get(shape, (DEFAULT_CMAP, DEFAULT_CMAP))
    return pair[0]


def _resolve_stream_cmap(cmap_arg, meta, output_dir=''):
    if cmap_arg:
        return cmap_arg
    shape = _detect_shape(meta, output_dir)
    pair = CASE_CMAPS.get(shape, (DEFAULT_CMAP, DEFAULT_CMAP))
    return pair[1]


# ---------------------------------------------------------------------------
# Obstacle overlay helper
# ---------------------------------------------------------------------------
OBSTACLE_COLOR = '#000000'  # strict black

def _overlay_obstacles(ax, obstacle_mask):
    """Draw obstacle regions as strict black overlay."""
    if obstacle_mask is None or obstacle_mask.size == 0:
        return
    # Create a masked array: show obstacles as black
    obs = np.ma.masked_where(~obstacle_mask, np.ones_like(obstacle_mask, dtype=float))
    ax.imshow(obs, origin='lower', cmap='gray_r', aspect='auto',
              vmin=0, vmax=1, alpha=1.0,
              interpolation='nearest')


# ---------------------------------------------------------------------------
# PNG rendering
# ---------------------------------------------------------------------------
def render_contour(ax, vel, cmap, vmin, vmax, obstacle=None):
    im = ax.imshow(vel, origin='lower', cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='bilinear')
    _overlay_obstacles(ax, obstacle)
    ax.axis('off')
    ax.set_facecolor('white')
    return im


def render_streamlines(ax, u, v, cmap, obstacle=None, density=1.0):
    ny, nx_grid = u.shape
    yg, xg = np.mgrid[0:ny, 0:nx_grid]
    step = max(1, nx_grid // 50)
    vel_mag = np.sqrt(u[::step, ::step]**2 + v[::step, ::step]**2)
    sp = ax.streamplot(xg[::step, ::step], yg[::step, ::step],
                       u[::step, ::step], v[::step, ::step],
                       color=vel_mag,
                       cmap=cmap, density=density, linewidth=0.8, arrowsize=0.8)
    _overlay_obstacles(ax, obstacle)
    ax.axis('off')
    ax.set_facecolor('white')
    return sp, vel_mag


def save_png_combined(data, output_dir, frame, cmap_contour, cmap_stream, field='velocity'):
    vel = np.array(data[field])
    u = np.array(data['u'])
    v = np.array(data['v'])
    obs = np.array(data.get('obstacle', []))
    if obs.ndim == 1 and obs.size > 0:
        obs = obs.reshape(data['ny'], data['nx'])
    if vel.ndim == 1:
        vel = vel.reshape(data['ny'], data['nx'])
    if u.ndim == 1:
        u = u.reshape(data['ny'], data['nx'])
    if v.ndim == 1:
        v = v.reshape(data['ny'], data['nx'])

    field_label = 'Pressure' if field == 'rho' else 'Velocity Magnitude'
    vmax_val = max(vel.max(), 0.01)
    vmin_val = vel.min() if field == 'rho' else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('white')

    im = render_contour(ax1, vel, cmap_contour, vmin_val, vmax_val, obs)
    cbar1 = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar1.set_label('Velocity Magnitude (m/s)', color='black')

    sp, smag = render_streamlines(ax2, u, v, cmap_stream, obs)
    if sp and smag.size > 0:
        cbar2 = plt.colorbar(sp.lines, ax=ax2, shrink=0.8)
        cbar2.set_label('Velocity Magnitude (m/s)', color='black')

    plt.tight_layout()
    path = os.path.join(output_dir, f'frame_{int(frame):04d}.png')
    plt.savefig(path, dpi=120, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def save_png_split(data, output_dir, frame, cmap_contour, cmap_stream, field='velocity'):
    vel = np.array(data[field])
    u = np.array(data['u'])
    v = np.array(data['v'])
    obs = np.array(data.get('obstacle', []))
    if obs.ndim == 1 and obs.size > 0:
        obs = obs.reshape(data['ny'], data['nx'])
    if vel.ndim == 1:
        vel = vel.reshape(data['ny'], data['nx'])
    if u.ndim == 1:
        u = u.reshape(data['ny'], data['nx'])
    if v.ndim == 1:
        v = v.reshape(data['ny'], data['nx'])

    field_label = 'Pressure' if field == 'rho' else 'Velocity Magnitude'
    vmax_val = max(vel.max(), 0.01)
    vmin_val = vel.min() if field == 'rho' else 0

    # Contour
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor('white')
    im = render_contour(ax, vel, cmap_contour, vmin_val, vmax_val, obs)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Velocity Magnitude (m/s)', color='black')
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(right=0.92)
    path = os.path.join(output_dir, f'contour_{int(frame):04d}.png')
    plt.savefig(path, dpi=120, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Streamlines
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor('white')
    sp, smag = render_streamlines(ax, u, v, cmap_stream, obs)
    if sp and smag.size > 0:
        cbar = plt.colorbar(sp.lines, ax=ax, shrink=0.8)
        cbar.set_label('Velocity Magnitude (m/s)', color='black')
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(right=0.92)
    path = os.path.join(output_dir, f'streamlines_{int(frame):04d}.png')
    plt.savefig(path, dpi=120, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Video overlay rendering (contour + streamlines on same axes)
# ---------------------------------------------------------------------------
def render_video_overlay(data, output_dir, frame, cmap, field='velocity'):
    vel = np.array(data[field])
    u = np.array(data['u'])
    v = np.array(data['v'])
    obs = np.array(data.get('obstacle', []))
    if obs.ndim == 1 and obs.size > 0:
        obs = obs.reshape(data['ny'], data['nx'])
    if vel.ndim == 1:
        vel = vel.reshape(data['ny'], data['nx'])
    if u.ndim == 1:
        u = u.reshape(data['ny'], data['nx'])
    if v.ndim == 1:
        v = v.reshape(data['ny'], data['nx'])

    vmax_val = max(vel.max(), 0.01)
    vmin_val = 0

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor('white')

    im = ax.imshow(vel, origin='lower', cmap=cmap, aspect='auto',
                   vmin=vmin_val, vmax=vmax_val, interpolation='bilinear')
    _overlay_obstacles(ax, obs)

    ny, nx_grid = u.shape
    yg, xg = np.mgrid[0:ny, 0:nx_grid]
    step = max(1, nx_grid // 50)
    vel_mag = np.sqrt(u[::step, ::step]**2 + v[::step, ::step]**2)
    ax.streamplot(xg[::step, ::step], yg[::step, ::step],
                  u[::step, ::step], v[::step, ::step],
                  color=vel_mag,
                  cmap=cmap, density=1.0, linewidth=0.8, arrowsize=0.8)

    ax.axis('off')
    ax.set_facecolor('white')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Velocity Magnitude (m/s)', color='black')

    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(right=0.92)
    path = os.path.join(output_dir, f'vid_{int(frame):04d}.png')
    plt.savefig(path, dpi=120, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()


def make_video(output_dir):
    vid_pattern = os.path.join(output_dir, 'vid_*.png')
    output_path = os.path.join(output_dir, 'simulation.mp4')
    cmd = [
        'ffmpeg', '-y', '-framerate', '15',
        '-pattern_type', 'glob', '-i', vid_pattern,
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'medium', '-crf', '18',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Video saved: {output_path}")
        # Clean up individual frames
        for f in Path(output_dir).glob('vid_*.png'):
            f.unlink()
    else:
        print(f"  ffmpeg error: {result.stderr}")


# ---------------------------------------------------------------------------
# Strouhal computation (Welch FFT on forces.jsonl)
# ---------------------------------------------------------------------------
def compute_strouhal(output_dir):
    forces_path = os.path.join(output_dir, 'forces.jsonl')
    if not os.path.exists(forces_path):
        print("  No forces.jsonl found, skipping Strouhal")
        return None

    meta = _load_meta(output_dir)
    re_val = meta.get('re', 0)
    u_inflow = meta.get('u_inflow', 0.1)
    length_scale = meta.get('length_scale', 1.0)

    steps, cl_vals = [], []
    with open(forces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            steps.append(rec['step'])
            cl_vals.append(rec['cl'])

    cl = np.array(cl_vals)
    n = len(cl)

    trim = n // 5
    cl_trimmed = cl[trim:]
    steps_trimmed = np.array(steps[trim:])

    if len(cl_trimmed) < 64:
        print(f"  Too few samples ({len(cl_trimmed)}) for Welch FFT, skipping")
        return None

    if not HAS_SCIPY:
        print("  scipy not installed, skipping Strouhal")
        return None

    nperseg = min(len(cl_trimmed) // 4, 16384)
    if nperseg < 8:
        nperseg = len(cl_trimmed) // 4
    if nperseg < 4:
        return None

    f, psd = signal.welch(cl_trimmed - cl_trimmed.mean(),
                          fs=1.0,
                          nperseg=nperseg,
                          window='hann',
                          noverlap=nperseg // 2)

    if u_inflow > 0 and length_scale > 0:
        f_min = 0.01 * u_inflow / length_scale
        f_max = 0.5 * u_inflow / length_scale
    else:
        f_min, f_max = 1e-5, 0.1

    mask = (f >= f_min) & (f <= f_max)
    if not mask.any():
        print(f"  No peaks in frequency range [{f_min:.6f}, {f_max:.6f}]")
        return None

    f_band = f[mask]
    psd_band = psd[mask]
    idx_peak = np.argmax(psd_band)
    f_peak = f_band[idx_peak]

    st = f_peak * length_scale / u_inflow if u_inflow > 0 else 0

    print(f"  Strouhal = {st:.4f} (f_peak = {f_peak:.6f}, Re = {re_val})")

    meta['strouhal'] = st
    if len(cl_trimmed) > 100:
        meta['cl_amplitude'] = float(np.std(cl_trimmed))
        cd_vals = []
        with open(forces_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cd_vals.append(rec['cd'])
        meta['cd_mean'] = float(np.mean(np.array(cd_vals)[trim:]))

    meta_path = os.path.join(output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Updated {meta_path}")

    return st


# ---------------------------------------------------------------------------
# Vorticity rendering (RdBu symmetric colormap)
# ---------------------------------------------------------------------------
def save_vorticity_png(data, output_dir, frame):
    omega = np.array(data.get('omega', []))
    obs = np.array(data.get('obstacle', []))
    if omega.ndim == 0 or omega.size == 0:
        print(f"  No omega field in frame {frame}, skipping")
        return
    if omega.ndim == 1:
        omega = omega.reshape(data['ny'], data['nx'])
    if obs.ndim == 1 and obs.size > 0:
        obs = obs.reshape(data['ny'], data['nx'])

    # Symmetric limits around 0
    vmax = max(abs(omega.max()), abs(omega.min()), 1e-6)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor('white')
    im = render_contour(ax, omega, 'RdBu', -vmax, vmax, obs)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Vorticity (1/s)', color='black')
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(right=0.92)
    path = os.path.join(output_dir, f'vorticity_{int(frame):04d}.png')
    plt.savefig(path, dpi=120, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='LBM-2D Post-Processor')
    parser.add_argument('input_dir', help='Output directory (e.g. output/re100)')
    parser.add_argument('--split', action='store_true',
                        help='Render contour and streamlines as separate PNGs')
    parser.add_argument('--cmap', default=None,
                        help='Colormap override (jet, coolwarm, viridis, plasma, RdBu)')
    parser.add_argument('--last-only', action='store_true',
                        help='Only render the last frame')
    parser.add_argument('--strouhal', action='store_true',
                        help='Compute Strouhal from forces.jsonl')
    parser.add_argument('--vorticity', action='store_true',
                        help='Render vorticity (omega) field')
    parser.add_argument('--field', default='velocity',
                        choices=['velocity', 'pressure'],
                        help='Field to render as contour (velocity or pressure)')
    parser.add_argument('--friction', action='store_true',
                        help='Print friction factor from meta.json (ribbed channel)')
    parser.add_argument('--video', action='store_true',
                        help='Render overlay video (contour + streamlines on same frame)')
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    meta = _load_meta(input_dir)
    frame_files = _list_frames(input_dir)

    # Map --field to data key
    field_key = 'rho' if args.field == 'pressure' else 'velocity'

    if frame_files:
        cmap_primary = _resolve_cmap(args.cmap, meta, input_dir)
        cmap_stream = _resolve_stream_cmap(args.cmap, meta, input_dir)
        print(f"Colormap: contour={cmap_primary}, streamlines={cmap_stream}, field={args.field}")

        if args.last_only:
            frame_files = frame_files[-1:]

        for vtk_path in frame_files:
            frame_match = re.search(r'frame_(\d+)', vtk_path.name)
            frame_num = int(frame_match.group(1)) if frame_match else 0
            data = _load_frame(str(vtk_path))

            if not HAS_MPL:
                print("matplotlib not installed, skipping PNG output")
            elif args.vorticity:
                save_vorticity_png(data, input_dir, frame_num)
            elif args.video:
                pass  # rendered in video section below
            elif args.split:
                save_png_split(data, input_dir, frame_num, cmap_primary, cmap_stream, field_key)
            else:
                save_png_combined(data, input_dir, frame_num, cmap_primary, cmap_stream, field_key)
    else:
        print(f"No frame JSON files found in {input_dir}/frames/")

    if args.video and frame_files:
        cmap_primary = _resolve_cmap(args.cmap, meta, input_dir)
        print(f"  Rendering {len(frame_files)} video frames with colormap={cmap_primary}")
        for vtk_path in frame_files:
            frame_match = re.search(r'frame_(\d+)', vtk_path.name)
            frame_num = int(frame_match.group(1)) if frame_match else 0
            data = _load_frame(str(vtk_path))
            if HAS_MPL:
                render_video_overlay(data, input_dir, frame_num, cmap_primary, field_key)
        make_video(input_dir)

    if args.strouhal:
        compute_strouhal(input_dir)

    if args.friction:
        f = meta.get('friction_factor', None)
        f_smooth = meta.get('f_smooth', None)
        ratio = meta.get('f_ratio', None)
        xr_h = meta.get('xr_h', None)
        u_bulk = meta.get('u_bulk', None)
        if f is not None:
            print(f"  Friction factor f = {f:.4f}")
            if f_smooth:
                print(f"  Smooth channel f_smooth = {f_smooth:.4f}")
            if ratio:
                print(f"  f/f_smooth = {ratio:.2f}")
            if xr_h:
                print(f"  Xr/h = {xr_h:.2f}")
            if u_bulk:
                print(f"  u_bulk = {u_bulk:.6f}")
        else:
            print("  No friction factor data in meta.json")

    print("Done.")


if __name__ == '__main__':
    main()
