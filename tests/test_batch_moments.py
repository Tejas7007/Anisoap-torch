from __future__ import annotations

import numpy as np
from anisoap_rust_lib import compute_moments, compute_moments_batch


def test_batched_moments_match_single_pair_implementation() -> None:
    rng = np.random.default_rng(11)
    pair_count = 12
    max_degree = 6

    factors = rng.normal(size=(pair_count, 3, 3))
    matrices = np.einsum("nij,nkj->nik", factors, factors)
    matrices += np.eye(3)[None, :, :] * 0.75
    centers = rng.normal(size=(pair_count, 3))

    flattened = np.asarray(
        compute_moments_batch(matrices.reshape(pair_count, 9), centers, max_degree)
    )
    batched = flattened.reshape(pair_count, max_degree + 1, max_degree + 1, max_degree + 1)

    for index in range(pair_count):
        expected = np.asarray(compute_moments(matrices[index], centers[index], max_degree))
        np.testing.assert_allclose(batched[index], expected, rtol=1e-12, atol=1e-12)
