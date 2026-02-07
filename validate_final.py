"""
Validation script for AniSOAP PyTorch optimization.
Checks all output blocks for NaNs, Infs, and reports shapes/magnitudes.

Usage:
    python validate_final.py
"""
import numpy as np
from anisoap.representations import EllipsoidalDensityProjection
from ase.io import read
import time

HYPERS = {
    "max_angular": 6,
    "radial_basis_name": "gto",
    "cutoff_radius": 7.0,
    "radial_gaussian_width": 1.5,
    "rotation_type": "quaternion",
    "rotation_key": "c_q",
}

for name, path in [("ellipsoids", "notebooks/ellipsoids.xyz"), ("benzenes", "notebooks/benzenes.xyz")]:
    frames = read(path, ":")[:50]
    for frame in frames:
        if "c_diameter[1]" not in frame.arrays:
            n = len(frame)
            frame.arrays["c_diameter[1]"] = np.ones(n) * 3.0
            frame.arrays["c_diameter[2]"] = np.ones(n) * 3.0
            frame.arrays["c_diameter[3]"] = np.ones(n) * 1.0
        if "c_q" not in frame.arrays:
            n = len(frame)
            frame.arrays["c_q"] = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))

    edp = EllipsoidalDensityProjection(**HYPERS)

    # Run 3 times and take best
    times = []
    for _ in range(3):
        start = time.perf_counter()
        result = edp.transform(frames)
        times.append(time.perf_counter() - start)

    print(f"\n{'='*60}")
    print(f"SYSTEM: {name}")
    print(f"  best={min(times):.3f}s  mean={np.mean(times):.3f}s  (3 runs)")
    print(f"{'='*60}")

    # Check output is valid
    n_blocks = len(result.keys)
    all_valid = True
    for key in result.keys:
        block = result.block(key)
        vals = np.array(block.values)
        has_nan = np.any(np.isnan(vals))
        has_inf = np.any(np.isinf(vals))
        if has_nan or has_inf:
            all_valid = False
        print(f"  Block {tuple(key)}: shape={vals.shape}, "
              f"has_nan={has_nan}, has_inf={has_inf}, "
              f"max_abs={np.max(np.abs(vals)):.4f}")

    print(f"\n  Total blocks: {n_blocks}")
    print(f"  All valid (no NaN/Inf): {all_valid}")
