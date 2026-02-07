"""
Benchmarking script for AniSOAP PyTorch optimization.
Profiles both ellipsoids and benzenes systems with cProfile.

Usage:
    python profile_baseline.py
"""
import cProfile
import pstats
import time
import numpy as np
from anisoap.representations import EllipsoidalDensityProjection
from ase.io import read

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

    # Ensure frames have required arrays
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

    start = time.perf_counter()
    profiler = cProfile.Profile()
    profiler.enable()
    edp.transform(frames)
    profiler.disable()
    wall = time.perf_counter() - start

    print(f"\n{'='*60}")
    print(f"SYSTEM: {name} | Wall time: {wall:.3f}s")
    print(f"{'='*60}")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    stats.dump_stats(f"baseline_{name}.prof")
    print(f"Saved: baseline_{name}.prof")
