"""Validate finite AniSOAP-Torch descriptor outputs on upstream example data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from ase.io import read

from anisoap.representations import EllipsoidalDensityProjection

HYPERS = {
    "max_angular": 6,
    "radial_basis_name": "gto",
    "cutoff_radius": 7.0,
    "radial_gaussian_width": 1.5,
    "rotation_type": "quaternion",
    "rotation_key": "c_q",
}


def prepare_frames(path: Path, limit: int):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing validation data at {path}. Run 'bash scripts/bootstrap.sh' first."
        )

    frames = read(path, ":")[:limit]
    if not frames:
        raise ValueError(f"No frames were loaded from {path}")

    for frame in frames:
        atom_count = len(frame)
        frame.arrays.setdefault("c_diameter[1]", np.full(atom_count, 3.0))
        frame.arrays.setdefault("c_diameter[2]", np.full(atom_count, 3.0))
        frame.arrays.setdefault("c_diameter[3]", np.full(atom_count, 1.0))
        frame.arrays.setdefault(
            "c_q", np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (atom_count, 1))
        )

    return frames


def validate_workload(name: str, path: Path, frame_limit: int) -> None:
    frames = prepare_frames(path, frame_limit)
    projection = EllipsoidalDensityProjection(**HYPERS)

    start = time.perf_counter()
    result = projection.transform(frames)
    duration = time.perf_counter() - start

    block_count = 0
    value_count = 0
    max_abs = 0.0

    for key in result.keys:
        values = np.asarray(result.block(key).values)
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"Non-finite values detected in {name} block {tuple(key)}")
        block_count += 1
        value_count += values.size
        if values.size:
            max_abs = max(max_abs, float(np.max(np.abs(values))))

    print(
        f"{name}: frames={len(frames)}, blocks={block_count}, values={value_count}, "
        f"max_abs={max_abs:.6g}, elapsed={duration:.4f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1, help="Maximum frames per workload")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".worktree/AniSOAP/notebooks"),
        help="Directory containing benzenes.xyz and ellipsoids.xyz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be positive")

    validate_workload("ellipsoids", args.data_root / "ellipsoids.xyz", args.frames)
    validate_workload("benzenes", args.data_root / "benzenes.xyz", args.frames)
    print("Validation passed: all descriptor values are finite.")


if __name__ == "__main__":
    main()
