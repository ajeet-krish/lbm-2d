#!/usr/bin/env python3
"""Extract force statistics from forces.jsonl for all configs."""

import json, os, sys
import numpy as np

PROJROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configs: (output_dir, label, is_normalized)
# is_normalized=True means forces.jsonl has proper Cd/Cl
# is_normalized=False means forces.jsonl has raw Fx/Fy

CONFIGS = [
    # Cylinder (proper Cd/Cl)
    ("output/cylinder/re100", "cylinder_re100", True),
    ("output/cylinder/re200", "cylinder_re200", True),
    # Flat plate (proper Cd/Cl)
    ("output/flatplate/re500_aoa0", "flatplate_re500_aoa0", True),
    ("output/flatplate/re1000_aoa-10", "flatplate_re1000_aoa-10", True),
    ("output/flatplate/re1000_aoa-5", "flatplate_re1000_aoa-5", True),
    ("output/flatplate/re1000_aoa0", "flatplate_re1000_aoa0", True),
    ("output/flatplate/re1000_aoa5", "flatplate_re1000_aoa5", True),
    ("output/flatplate/re1000_aoa10", "flatplate_re1000_aoa10", True),
    ("output/flatplate/re1000_aoa15", "flatplate_re1000_aoa15", True),
    ("output/flatplate/re2000_aoa0", "flatplate_re2000_aoa0", True),
    # Step (raw Fx/Fy)
    ("output/step/re100", "step_re100", False),
    ("output/step/re200", "step_re200", False),
    ("output/step/re400", "step_re400", False),
    # Square cylinder (proper Cd/Cl)
    ("output/square_cylinder/re200", "square_cylinder_re200", True),
    # Orifice plate (raw Fx/Fy)
    ("output/orifice_plate/re100_1p1h", "orifice_1p1h", False),
    ("output/orifice_plate/re100_1p3h", "orifice_1p3h", False),
    ("output/orifice_plate/re100_2p", "orifice_2p", False),
    ("output/orifice_plate/re100_3p", "orifice_3p", False),
    # Periodic hills (raw Fx/Fy)
    ("output/periodic_hills/re100", "phills_re100", False),
    ("output/periodic_hills/re1000", "phills_re1000", False),
    ("output/periodic_hills/re2800", "phills_re2800", False),
    # Cylinder near wall (proper Cd/Cl)
    ("output/cylinder_near_wall/re100_gap10", "cnw_gap10", True),
    ("output/cylinder_near_wall/re100_gap20", "cnw_gap20", True),
    ("output/cylinder_near_wall/re100_gap40", "cnw_gap40", True),
    # Side-by-side (proper Cd/Cl)
    ("output/side_by_side/re100_sd20", "sbs_sd20", True),
    ("output/side_by_side/re100_sd30", "sbs_sd30", True),
    ("output/side_by_side/re100_sd50", "sbs_sd50", True),
    # Rotating cylinder (proper Cd/Cl)
    ("output/rotating_cylinder/re100_w5", "rot_w5", True),
    ("output/rotating_cylinder/re100_w10", "rot_w10", True),
    ("output/rotating_cylinder/re100_w20", "rot_w20", True),
    # Urban side (raw Fx/Fy)
    ("output/urban_side_ar3_re100", "urban_side_ar3", False),
    ("output/urban_side_ar5_re100", "urban_side_ar5", False),
    ("output/urban_side_ar6_3b_re100", "urban_side_ar6_3b", False),
    ("output/urban_side_ar8_re100", "urban_side_ar8", False),
    ("output/urban_topdown_re100", "urban_topdown", False),
    ("output/urban_topdown_h_re100", "urban_topdown_h", False),
    ("output/urban/downwash_re100", "urban_downwash", False),
    # Ribs (raw Fx/Fy)
    ("output/ribs/re50", "ribs_re50", False),
    ("output/ribs/re100", "ribs_re100", False),
    ("output/ribs/re200", "ribs_re200", False),
]


def load_forces(outdir):
    """Load forces.jsonl, return (steps, cd/cl or fx/fy) arrays."""
    path = os.path.join(outdir, "forces.jsonl")
    if not os.path.exists(path):
        return None, None, None
    steps, v1, v2 = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                steps.append(d.get("step", 0))
                v1.append(d.get("cd", 0.0))
                v2.append(d.get("cl", 0.0))
            except json.JSONDecodeError:
                continue
    if not steps:
        return None, None, None
    return np.array(steps), np.array(v1), np.array(v2)


def compute_strouhal(steps, fy, dt=1.0):
    """Compute Strouhal number from fy time series using FFT.
    
    Uses zero-crossing method for more robust frequency detection.
    Returns frequency in (1/lattice time units).
    """
    if len(fy) < 200:
        return None
    
    # Skip first 40% as transient (vortex shedding needs time to develop)
    n_skip = int(len(fy) * 0.4)
    if n_skip < 50:
        n_skip = 50
    fy_seg = fy[n_skip:]
    
    if len(fy_seg) < 100:
        return None
    
    # Zero-crossing method: count upward crossings of the mean
    fy_mean = np.mean(fy_seg)
    crossings = 0
    for i in range(1, len(fy_seg)):
        if fy_seg[i-1] < fy_mean and fy_seg[i] >= fy_mean:
            crossings += 1
    
    if crossings < 2:
        return None
    
    duration = len(fy_seg) * dt
    freq = crossings / (2.0 * duration)  # crossings per half-cycle, so /2 for full cycles
    return freq


def main():
    results = {}
    for outdir, label, is_norm in CONFIGS:
        fulldir = os.path.join(PROJROOT, outdir)
        steps, v1, v2 = load_forces(fulldir)
        if steps is None:
            print(f"{label}: NO FORCES DATA")
            continue

        # Skip first 20% as transient
        n_skip = int(len(v1) * 0.2)
        if n_skip < 10:
            n_skip = 0

        if is_norm:
            # v1=cd, v2=cl (properly normalized)
            cd_mean = np.mean(v1[n_skip:])
            cl_mean = np.mean(v2[n_skip:])
            cl_std = np.std(v2[n_skip:])
            st_raw = compute_strouhal(steps, v2)
            st = st_raw if st_raw else None
            st_str = f"{st:.4f}" if st is not None else "N/A"
            label_str = f"Cd={cd_mean:.3f}  Cl={cl_mean:.3f}  St={st_str}"
        else:
            # v1=fx_total, v2=fy_total (raw forces)
            fx_mean = np.mean(v1[n_skip:])
            fy_mean = np.mean(v2[n_skip:])
            fy_std = np.std(v2[n_skip:])
            st_raw = compute_strouhal(steps, v2)
            st = st_raw if st_raw else None
            cd_mean = fx_mean  # alias for reporting
            cl_mean = fy_mean
            st_str = f"{st:.4f}" if st is not None else "N/A"
            label_str = f"Fx={fx_mean:.2f}  Fy={fy_mean:.2f}  St={st_str}"

        n_steps = int(steps[-1]) if len(steps) > 0 else 0

        results[label] = {
            "is_normalized": is_norm,
            "cd": float(cd_mean),
            "cl": float(cl_mean),
            "st": float(st) if st is not None else None,
            "n_steps": n_steps,
        }

        print(f"{label}: {label_str}  steps={n_steps}")

    # Save results
    outpath = os.path.join(PROJROOT, "output", "force_stats.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
