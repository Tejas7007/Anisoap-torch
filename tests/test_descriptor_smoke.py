from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from anisoap.representations import EllipsoidalDensityProjection


def test_one_frame_descriptor_is_finite() -> None:
    data_path = Path(".worktree/AniSOAP/notebooks/ellipsoids.xyz")
    if not data_path.is_file():
        pytest.skip("Run scripts/bootstrap.sh to create the pinned upstream worktree")

    frames = read(data_path, ":1")
    for frame in frames:
        atom_count = len(frame)
        frame.arrays.setdefault("c_diameter[1]", np.full(atom_count, 3.0))
        frame.arrays.setdefault("c_diameter[2]", np.full(atom_count, 3.0))
        frame.arrays.setdefault("c_diameter[3]", np.full(atom_count, 1.0))
        frame.arrays.setdefault(
            "c_q", np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (atom_count, 1))
        )

    projection = EllipsoidalDensityProjection(
        max_angular=3,
        radial_basis_name="gto",
        cutoff_radius=7.0,
        radial_gaussian_width=1.5,
        rotation_type="quaternion",
        rotation_key="c_q",
    )
    result = projection.transform(frames)

    values = [np.asarray(block.values) for block in result.blocks()]
    assert values
    assert all(np.all(np.isfinite(block_values)) for block_values in values)
