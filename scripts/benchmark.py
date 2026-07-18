"""Run the optimized AniSOAP descriptor benchmark on upstream example data."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
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
            f"Missing benchmark data at {path}. Run 'bash scripts/bootstrap.sh' first."
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


def run_workload(name: str, path: Path, frame_limit: int, repeats: int, profile: bool):
    frames = prepare_frames(path, frame_limit)
    projection = EllipsoidalDensityProjection(**HYPERS)

    projection.transform(frames)
    durations: list[float] = []

    for repeat in range(repeats):
        profiler = cProfile.Profile() if profile else None
        if profiler is not None:
            profiler.enable()

        start = time.perf_counter()
        projection.transform(frames)
        durations.append(time.perf_counter() - start)

        if profiler is not None:
            profiler.disable()
            profile_path = Path(f"{name}-run-{repeat + 1}.prof")
            profiler.dump_stats(profile_path)
            pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)
            print(f"Profile written to {profile_path}")

    return {
        "name": name,
        "frames": len(frames),
        "repeats": repeats,
        "best_seconds": min(durations),
        "mean_seconds": float(np.mean(durations)),
        "all_seconds": durations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=50, help="Maximum frames per workload")
    parser.add_argument("--repeats", type=int, default=3, help="Measured runs after warm-up")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".worktree/AniSOAP/notebooks"),
        help="Directory containing benzenes.xyz and ellipsoids.xyz",
    )
    parser.add_argument("--profile", action="store_true", help="Write cProfile output")
    parser.add_argument("--json", type=Path, help="Optional path for JSON results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.repeats < 1:
        raise SystemExit("--frames and --repeats must both be positive")

    workloads = [
        ("ellipsoids", args.data_root / "ellipsoids.xyz"),
        ("benzenes", args.data_root / "benzenes.xyz"),
    ]
    results = [
        run_workload(name, path, args.frames, args.repeats, args.profile)
        for name, path in workloads
    ]

    for result in results:
        print(
            f"{result['name']}: best={result['best_seconds']:.4f}s, "
            f"mean={result['mean_seconds']:.4f}s, frames={result['frames']}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"workloads": results}, indent=2) + "\n")
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
