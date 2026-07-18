from __future__ import annotations

import numpy as np

from anisoap.representations.radial_basis import GTORadialBasis, MonomialBasis


def random_rotations(rng: np.random.Generator, count: int) -> np.ndarray:
    rotations = []
    for _ in range(count):
        matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(matrix) < 0:
            matrix[:, 0] *= -1
        rotations.append(matrix)
    return np.asarray(rotations)


def assert_batch_matches_single(basis) -> None:
    rng = np.random.default_rng(7)
    count = 24
    displacements = rng.normal(size=(count, 3))
    lengths = rng.uniform(0.5, 2.5, size=(count, 3))
    rotations = random_rotations(rng, count)

    batch_precision, batch_center, batch_constant = basis.compute_gaussian_parameters_batch(
        displacements, lengths, rotations
    )

    for index in range(count):
        precision, center, constant = basis.compute_gaussian_parameters(
            displacements[index], lengths[index], rotations[index]
        )
        np.testing.assert_allclose(batch_precision[index], precision, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch_center[index], center, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch_constant[index], constant, rtol=1e-12, atol=1e-12)


def test_gto_batch_matches_single_pair_equations() -> None:
    basis = GTORadialBasis(
        max_angular=4,
        cutoff_radius=7.0,
        radial_gaussian_width=1.5,
    )
    assert_batch_matches_single(basis)


def test_monomial_batch_matches_single_pair_equations() -> None:
    basis = MonomialBasis(max_angular=4, cutoff_radius=7.0)
    assert_batch_matches_single(basis)
