#!/usr/bin/env python3
"""
Generate airfoil validation plots for the LBM-2D portfolio.
Reads forces.csv from each AoA simulation, produces Cl-vs-Alpha, Cd-vs-Alpha,
drag polar, and NACA 0012 vs 2412 comparison.

Usage:
    python3 scripts/plot_airfoil.py
"""
import sys, os, json, re
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available -- install with: pip install matplotlib")

# Project root (two levels up from scripts/)
ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "docs" / "assets" / "images"
DATA_DIR = ROOT / "output" / "airfoil"

# ------------------------------------------------------------------
# Load LBM data from forces.csv files
# ------------------------------------------------------------------
def load_lbm_data(series, Re, aoas):
    """Load force coefficients from saved simulations."""
    results = {'aoa': [], 'cd_mean': [], 'cl_mean': [], 'cd_std': [], 'cl_std': []}
    for aoa in aoas:
        dirname = f"naca{series:04d}_re{Re}_aoa{aoa}"
        csv_path = DATA_DIR / dirname / "forces.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping AoA={aoa}")
            continue
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        steps = data[:, 0]
        cd = data[:, 1]
        cl = data[:, 2]

        # Use last half for mean
        n = len(cd)
        start = n // 2
        cd_mean = np.mean(cd[start:])
        cl_mean = np.mean(cl[start:])
        cd_std = np.std(cd[start:])
        cl_std = np.std(cl[start:])

        results['aoa'].append(aoa)
        results['cd_mean'].append(cd_mean)
        results['cl_mean'].append(cl_mean)
        results['cd_std'].append(cd_std)
        results['cl_std'].append(cl_std)

        print(f"  AoA={aoa:2d}  Cd={cd_mean:.4f} +/- {cd_std:.4f}  Cl={cl_mean:.4f} +/- {cl_std:.4f}")

    return results

# ------------------------------------------------------------------
# Reference data (thin airfoil theory)
# ------------------------------------------------------------------
def thin_airfoil_theory(aoas_deg):
    aoas_rad = np.radians(aoas_deg)
    return 2.0 * np.pi * aoas_rad  # Cl = 2*pi*alpha

# ------------------------------------------------------------------
# Plot: Cl vs AoA
# ------------------------------------------------------------------
def plot_cl_vs_aoa(lbm_0012, lbm_2412, ref_data, filename="airfoil_cl_vs_aoa.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    # LBM NACA 0012
    ax.errorbar(lbm_0012['aoa'], lbm_0012['cl_mean'], yerr=lbm_0012['cl_std'],
                fmt='o-', color='#00d4ff', capsize=3, label='LBM NACA 0012 (Re=1000)')
    # LBM NACA 2412
    ax.errorbar(lbm_2412['aoa'], lbm_2412['cl_mean'], yerr=lbm_2412['cl_std'],
                fmt='s--', color='#00f5d4', capsize=3, label='LBM NACA 2412 (Re=1000)')

    # Thin airfoil theory
    aoas_fine = np.linspace(0, 16, 100)
    ax.plot(aoas_fine, thin_airfoil_theory(aoas_fine), '--', color='#ff6b35',
            alpha=0.7, label='Thin airfoil theory (2$\pi$ per rad)')

    # Reference data from SU2 RANS (Re=1e6) and Ladson (1988)
    if ref_data:
        ax.plot(ref_data['aoa'], ref_data['cl_su2'], '^-.', color='#39d353',
                label='SU2 RANS NACA 0012 (Re=1e6)')
        ax.plot(ref_data['aoa'], ref_data['cl_ladson'], 'v:', color='#8b949e',
                label='Ladson 1988 exp. NACA 0012 (Re=1e6)')

    ax.set_xlabel('Angle of Attack (deg)', color='#c9d1d9')
    ax.set_ylabel('Lift Coefficient $C_l$', color='#c9d1d9')
    ax.set_title('Lift Coefficient vs Angle of Attack', color='#c9d1d9')
    ax.legend(framealpha=0.3, facecolor='#161b22', labelcolor='#c9d1d9',
              edgecolor='#21262d')
    ax.tick_params(colors='#8b949e')
    ax.spines['bottom'].set_color('#21262d')
    ax.spines['top'].set_color('#21262d')
    ax.spines['left'].set_color('#21262d')
    ax.spines['right'].set_color('#21262d')
    ax.grid(True, alpha=0.15, color='#484f58')
    ax.set_xlim(0, 17)

    plt.tight_layout()
    path = IMG_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

# ------------------------------------------------------------------
# Plot: Cd vs AoA
# ------------------------------------------------------------------
def plot_cd_vs_aoa(lbm_0012, lbm_2412, filename="airfoil_cd_vs_aoa.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    ax.errorbar(lbm_0012['aoa'], lbm_0012['cd_mean'], yerr=lbm_0012['cd_std'],
                fmt='o-', color='#00d4ff', capsize=3, label='LBM NACA 0012 (Re=1000)')
    ax.errorbar(lbm_2412['aoa'], lbm_2412['cd_mean'], yerr=lbm_2412['cd_std'],
                fmt='s--', color='#00f5d4', capsize=3, label='LBM NACA 2412 (Re=1000)')

    ax.set_xlabel('Angle of Attack (deg)', color='#c9d1d9')
    ax.set_ylabel('Drag Coefficient $C_d$', color='#c9d1d9')
    ax.set_title('Drag Coefficient vs Angle of Attack', color='#c9d1d9')
    ax.legend(framealpha=0.3, facecolor='#161b22', labelcolor='#c9d1d9',
              edgecolor='#21262d')
    ax.tick_params(colors='#8b949e')
    ax.spines['bottom'].set_color('#21262d')
    ax.spines['top'].set_color('#21262d')
    ax.spines['left'].set_color('#21262d')
    ax.spines['right'].set_color('#21262d')
    ax.grid(True, alpha=0.15, color='#484f58')
    ax.set_xlim(0, 17)

    plt.tight_layout()
    path = IMG_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

# ------------------------------------------------------------------
# Plot: Drag polar (Cd vs Cl)
# ------------------------------------------------------------------
def plot_drag_polar(lbm_0012, filename="airfoil_drag_polar.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    ax.plot(lbm_0012['cd_mean'], lbm_0012['cl_mean'], 'o-', color='#00d4ff',
            label='LBM NACA 0012 (Re=1000)')

    # Annotate with AoA
    for i, aoa in enumerate(lbm_0012['aoa']):
        ax.annotate(f'{aoa}°', (lbm_0012['cd_mean'][i], lbm_0012['cl_mean'][i]),
                    textcoords="offset points", xytext=(8, 5), color='#c9d1d9',
                    fontsize=9)

    ax.set_xlabel('Drag Coefficient $C_d$', color='#c9d1d9')
    ax.set_ylabel('Lift Coefficient $C_l$', color='#c9d1d9')
    ax.set_title('Drag Polar: NACA 0012 at Re=1000', color='#c9d1d9')
    ax.legend(framealpha=0.3, facecolor='#161b22', labelcolor='#c9d1d9',
              edgecolor='#21262d')
    ax.tick_params(colors='#8b949e')
    ax.spines['bottom'].set_color('#21262d')
    ax.spines['top'].set_color('#21262d')
    ax.spines['left'].set_color('#21262d')
    ax.spines['right'].set_color('#21262d')
    ax.grid(True, alpha=0.15, color='#484f58')
    ax.set_xlim(0, 0.35)

    plt.tight_layout()
    path = IMG_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    aoas_0012 = [0, 4, 8, 12, 16]
    aoas_2412 = [0, 4, 8]

    print("Loading NACA 0012 data...")
    lbm_0012 = load_lbm_data(12, 1000, aoas_0012)
    print("Loading NACA 2412 data...")
    lbm_2412 = load_lbm_data(2412, 1000, aoas_2412)

    # Reference SU2 data
    ref_aoas = [0, 4, 8, 12, 16]
    ref_data = {
        'aoa': ref_aoas,
        'cl_su2': [0.0014, 0.4453, 0.8718, 1.2647, 1.6082],
        'cd_su2': [0.0749, 0.0969, 0.1644, 0.2764, 0.4314],
        'cl_ladson': [0.000, 0.430, 0.840, 1.140, 1.100],
    }

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_cl_vs_aoa(lbm_0012, lbm_2412, ref_data)
    plot_cd_vs_aoa(lbm_0012, lbm_2412)
    plot_drag_polar(lbm_0012)

    # Also save data as JSON for the web page
    json_data = {
        'lbm_0012': {
            'aoa': lbm_0012['aoa'],
            'cd_mean': [round(v, 4) for v in lbm_0012['cd_mean']],
            'cl_mean': [round(v, 4) for v in lbm_0012['cl_mean']],
            'cd_std': [round(v, 4) for v in lbm_0012['cd_std']],
            'cl_std': [round(v, 4) for v in lbm_0012['cl_std']],
        },
        'lbm_2412': {
            'aoa': lbm_2412['aoa'],
            'cd_mean': [round(v, 4) for v in lbm_2412['cd_mean']],
            'cl_mean': [round(v, 4) for v in lbm_2412['cl_mean']],
            'cd_std': [round(v, 4) for v in lbm_2412['cd_std']],
            'cl_std': [round(v, 4) for v in lbm_2412['cl_std']],
        },
        'reference': {
            'aoa': ref_aoas,
            'cl_su2': ref_data['cl_su2'],
            'cd_su2': ref_data['cd_su2'],
            'cl_ladson': ref_data['cl_ladson'],
        }
    }
    json_path = IMG_DIR / "airfoil_data.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved {json_path}")

    print("Done.")

if __name__ == '__main__':
    main()
