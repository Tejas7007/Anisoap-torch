import warnings

import numpy as np
import torch
from metatensor import TensorMap
from scipy.special import gamma


def inverse_matrix_sqrt(matrix, rcond=1e-8, tol=1e-3):
    r"""Returns the inverse matrix square root using PyTorch.

    Parameters
    ----------
    matrix : torch.Tensor or np.ndarray
        Symmetric square matrix
    rcond : float
        Lower bound for eigenvalues
    tol : float
        Tolerance for reconstruction check

    Returns
    -------
    torch.Tensor
        S^{-1/2}
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)

    if not torch.allclose(matrix, matrix.T):
        raise ValueError("Matrix is not hermitian")

    eva, eve = torch.linalg.eigh(matrix)
    mask = eva > rcond
    eve = eve[:, mask]
    eva = eva[mask]

    result = eve @ torch.diag(1.0 / torch.sqrt(eva)) @ eve.T

    # Verification
    matrix2 = torch.linalg.pinv(result @ result)
    if torch.linalg.norm(matrix - matrix2).item() > tol:
        raise ValueError(
            f"Incurred Numerical Imprecision {torch.linalg.norm(matrix - matrix2).item():.8f}"
        )
    return result


def gto_square_norm(n, sigma):
    r"""Compute the square norm of GTOs."""
    return 0.5 * sigma ** (2 * n + 3) * gamma(n + 1.5)


def gto_prefactor(n, sigma):
    """Computes the normalization prefactor of an unnormalized GTO."""
    return np.sqrt(1 / gto_square_norm(n, sigma))


def gto_overlap(n, m, sigma_n, sigma_m):
    r"""Compute overlap of two *normalized* GTOs."""
    N_n = gto_prefactor(n, sigma_n)
    N_m = gto_prefactor(m, sigma_m)
    n_eff = (n + m) / 2
    sigma_eff = np.sqrt(2 * sigma_n**2 * sigma_m**2 / (sigma_n**2 + sigma_m**2))
    return N_n * N_m * gto_square_norm(n_eff, sigma_eff)


def monomial_square_norm(n, r_cut):
    """Compute the square norm of monomials."""
    return r_cut ** (2 * n + 3) / (2 * n + 3)


def monomial_prefactor(n, r_cut):
    """Computes the normalization prefactor of an unnormalized monomial basis."""
    return np.sqrt(1 / monomial_square_norm(n, r_cut))


def monomial_overlap(n, m, r_cut):
    r"""Compute overlap of two *normalized* monomials."""
    N_n = monomial_prefactor(n, r_cut)
    N_m = monomial_prefactor(m, r_cut)
    n_eff = (n + m) / 2
    return N_n * N_m * monomial_square_norm(n_eff, r_cut)


class _RadialBasis:
    """Base class for radial basis functions, with PyTorch backend."""

    def __init__(
        self,
        radial_basis,
        max_angular,
        cutoff_radius,
        max_radial=None,
        rcond=1e-8,
        tol=1e-3,
        device=None,
    ):
        self.radial_basis = radial_basis
        self.max_angular = max_angular
        self.cutoff_radius = cutoff_radius
        self.rcond = rcond
        self.tol = tol
        self.device = torch.device(device or "cpu")

        if type(cutoff_radius) == int:
            raise ValueError(
                "r_cut is set as an integer, which could cause overflow errors. Pass in float"
            )

        self.num_radial_functions = []
        for l in range(max_angular + 1):
            if max_radial is None:
                num_n = (max_angular - l) // 2 + 1
                self.num_radial_functions.append(num_n)
            elif isinstance(max_radial, list):
                if len(max_radial) <= l:
                    raise ValueError(
                        "If you specify a list of number of radial components, this list must be of length {}. Received {}.".format(
                            max_angular + 1, len(max_radial)
                        )
                    )
                if not isinstance(max_radial[l], int):
                    raise ValueError("`max_radial` must be None, int, or list of int")
                self.num_radial_functions.append(max_radial[l] + 1)
            elif isinstance(max_radial, int):
                self.num_radial_functions.append(max_radial + 1)
            else:
                raise ValueError("`max_radial` must be None, int, or list of int")
            if type(cutoff_radius) == int:
                raise ValueError(
                    "r_cut is set as an integer, which could cause overflow errors. Pass in float"
                )

    def get_num_radial_functions(self):
        return self.num_radial_functions

    def plot_basis(self, n_r=100):
        from matplotlib import pyplot as plt
        rs = np.linspace(0, self.cutoff_radius, n_r)
        plt.plot(rs, self.get_basis(rs))


class MonomialBasis(_RadialBasis):
    r"""Monomial basis with PyTorch backend."""

    def __init__(
        self,
        max_angular,
        cutoff_radius,
        max_radial=None,
        rcond=1e-8,
        tol=1e-3,
        device=None,
    ):
        super().__init__("monomial", max_angular, cutoff_radius, max_radial, rcond, tol, device)
        self.overlap_matrix = self.calc_overlap_matrix()

    def compute_gaussian_parameters(self, r_ij, lengths, rotation_matrix):
        center = r_ij
        diag = np.diag(1 / lengths**2)
        precision = rotation_matrix @ diag @ rotation_matrix.T
        constant = 0
        return precision, center, constant

    def compute_gaussian_parameters_batch(self, r_ij_all, lengths_all, rot_all):
        """Batched version using PyTorch."""
        dev = self.device
        r_ij = torch.as_tensor(r_ij_all, dtype=torch.float64, device=dev)
        lengths = torch.as_tensor(lengths_all, dtype=torch.float64, device=dev)
        rot = torch.as_tensor(rot_all, dtype=torch.float64, device=dev)
        N = r_ij.shape[0]

        inv_len_sq = 1.0 / lengths**2
        diag = torch.zeros(N, 3, 3, dtype=torch.float64, device=dev)
        diag[:, 0, 0] = inv_len_sq[:, 0]
        diag[:, 1, 1] = inv_len_sq[:, 1]
        diag[:, 2, 2] = inv_len_sq[:, 2]

        precision = rot @ diag @ rot.transpose(-1, -2)
        constant = torch.zeros(N, dtype=torch.float64, device=dev)
        center = r_ij.clone()

        return (
            precision.cpu().numpy(),
            center.cpu().numpy(),
            constant.cpu().numpy(),
        )

    def calc_overlap_matrix(self):
        max_deg = np.max(
            np.arange(self.max_angular + 1) + 2 * np.array(self.num_radial_functions)
        )
        n_grid = np.arange(max_deg)
        S = monomial_overlap(
            n_grid[:, np.newaxis], n_grid[np.newaxis, :], self.cutoff_radius
        )
        return torch.from_numpy(S).to(self.device)

    def orthonormalize_basis(self, features: TensorMap):
        for label, block in features.items():
            neighbors = np.unique(block.properties["neighbor_types"])
            for neighbor in neighbors:
                l = label["angular_channel"]
                neighbor_mask = block.properties["neighbor_types"] == neighbor
                n_arr = block.properties["n"][neighbor_mask].flatten()
                l_2n_arr = l + 2 * n_arr

                prefactor_arr = monomial_prefactor(l_2n_arr, self.cutoff_radius)
                block.values[:, :, neighbor_mask] *= prefactor_arr

                overlap_slice = self.overlap_matrix[l_2n_arr, :][:, l_2n_arr]
                ortho_matrix = inverse_matrix_sqrt(overlap_slice, self.rcond, self.tol)
                # Convert to numpy for metatensor compatibility
                ortho_np = ortho_matrix.cpu().numpy() if isinstance(ortho_matrix, torch.Tensor) else ortho_matrix

                vals = torch.from_numpy(np.array(block.values[:, :, neighbor_mask]))
                ortho_t = torch.from_numpy(ortho_np) if isinstance(ortho_np, np.ndarray) else ortho_matrix
                result = torch.einsum("ijk,kl->ijl", vals, ortho_t)
                block.values[:, :, neighbor_mask] = result.numpy()

        return features

    def get_basis(self, rs):
        all_gs = np.empty(shape=(len(rs), 1))
        for l in range(0, self.max_angular):
            n_arr = np.arange(self.num_radial_functions[l])
            l_2n_arr = l + 2 * n_arr
            gs = np.array([(rs ** (2 * n + l)) for n in n_arr]).T
            prefactor_arr = monomial_prefactor(l_2n_arr, self.cutoff_radius)
            gs *= prefactor_arr
            overlap_slice = self.overlap_matrix[l_2n_arr, :][:, l_2n_arr]
            ortho_matrix = inverse_matrix_sqrt(overlap_slice, self.rcond, self.tol).numpy()
            gs = np.einsum("jk,kl->jl", gs, ortho_matrix)
            if all_gs is None:
                all_gs = gs.copy()
            all_gs = np.hstack((all_gs, gs))
        return all_gs[:, 1:]


class GTORadialBasis(_RadialBasis):
    """GTO basis with PyTorch backend."""

    def __init__(
        self,
        max_angular,
        cutoff_radius,
        *,
        radial_gaussian_width,
        max_radial=None,
        rcond=1e-8,
        tol=1e-3,
        device=None,
    ):
        super().__init__("gto", max_angular, cutoff_radius, max_radial, rcond, tol, device)
        self.radial_gaussian_width = radial_gaussian_width
        self.overlap_matrix = self.calc_overlap_matrix()

    def compute_gaussian_parameters(self, r_ij, lengths, rotation_matrix):
        center = r_ij
        diag = np.diag(1 / lengths**2)
        precision = rotation_matrix @ diag @ rotation_matrix.T
        sigma = self.radial_gaussian_width
        new_precision = precision + np.eye(3) / sigma**2
        new_center = center - 1 / sigma**2 * np.linalg.solve(new_precision, r_ij)
        constant = (
            1 / sigma**2 * r_ij @ np.linalg.solve(new_precision, precision @ r_ij)
        )
        return new_precision, new_center, constant

    def compute_gaussian_parameters_batch(self, r_ij_all, lengths_all, rot_all):
        """Batched version using PyTorch.

        Parameters
        ----------
        r_ij_all : np.ndarray of shape (N, 3)
        lengths_all : np.ndarray of shape (N, 3)
        rot_all : np.ndarray of shape (N, 3, 3)

        Returns
        -------
        new_precision : np.ndarray (N, 3, 3)
        new_center : np.ndarray (N, 3)
        constant : np.ndarray (N,)
        """
        dev = self.device
        sigma = self.radial_gaussian_width

        r_ij = torch.as_tensor(r_ij_all, dtype=torch.float64, device=dev)
        lengths = torch.as_tensor(lengths_all, dtype=torch.float64, device=dev)
        rot = torch.as_tensor(rot_all, dtype=torch.float64, device=dev)
        N = r_ij.shape[0]

        inv_len_sq = 1.0 / lengths**2  # (N, 3)

        # Build batch diagonal: (N, 3, 3)
        diag = torch.zeros(N, 3, 3, dtype=torch.float64, device=dev)
        diag[:, 0, 0] = inv_len_sq[:, 0]
        diag[:, 1, 1] = inv_len_sq[:, 1]
        diag[:, 2, 2] = inv_len_sq[:, 2]

        # precision: (N, 3, 3)
        precision = rot @ diag @ rot.transpose(-1, -2)

        # new_precision: (N, 3, 3)
        eye = torch.eye(3, dtype=torch.float64, device=dev).unsqueeze(0)
        new_precision = precision + eye / sigma**2

        # Batch solve
        solved_r = torch.linalg.solve(new_precision, r_ij.unsqueeze(-1)).squeeze(-1)

        pr = torch.einsum('nij,nj->ni', precision, r_ij)
        solved_pr = torch.linalg.solve(new_precision, pr.unsqueeze(-1)).squeeze(-1)

        new_center = r_ij - (1.0 / sigma**2) * solved_r
        constant = (1.0 / sigma**2) * torch.einsum('ni,ni->n', r_ij, solved_pr)

        return (
            new_precision.cpu().numpy(),
            new_center.cpu().numpy(),
            constant.cpu().numpy(),
        )

    def calc_overlap_matrix(self):
        max_deg = np.max(
            np.arange(self.max_angular + 1) + 2 * np.array(self.num_radial_functions)
        )
        n_grid = np.arange(max_deg)
        sigma = self.radial_gaussian_width
        sigma_grid = np.ones(max_deg) * sigma
        S = gto_overlap(
            n_grid[:, np.newaxis],
            n_grid[np.newaxis, :],
            sigma_grid[:, np.newaxis],
            sigma_grid[np.newaxis, :],
        )
        return torch.from_numpy(S).to(dtype=torch.float64, device=self.device)

    def orthonormalize_basis(self, features: TensorMap):
        for label, block in features.items():
            neighbors = np.unique(block.properties["neighbor_types"])
            for neighbor in neighbors:
                l = label["angular_channel"]
                neighbor_mask = block.properties["neighbor_types"] == neighbor
                n_arr = block.properties["n"][neighbor_mask].flatten()
                l_2n_arr = l + 2 * n_arr

                prefactor_arr = gto_prefactor(l_2n_arr, self.radial_gaussian_width)
                block.values[:, :, neighbor_mask] *= prefactor_arr

                gto_overlap_slice = self.overlap_matrix[l_2n_arr, :][:, l_2n_arr]
                ortho_matrix = inverse_matrix_sqrt(gto_overlap_slice, self.rcond, self.tol)

                vals = torch.from_numpy(np.array(block.values[:, :, neighbor_mask])).to(self.device)
                result = torch.einsum("ijk,kl->ijl", vals, ortho_matrix)
                block.values[:, :, neighbor_mask] = result.cpu().numpy()

        return features

    def get_basis(self, rs):
        from matplotlib import pyplot as plt
        all_gs = np.empty(shape=(len(rs), 1))
        for l in range(0, self.max_angular):
            n_arr = np.arange(self.num_radial_functions[l])
            l_2n_arr = l + 2 * n_arr
            gs = np.array(
                [
                    (rs ** (2 * n + l))
                    * np.exp(-(rs**2.0) / (2 * self.radial_gaussian_width**2.0))
                    for n in n_arr
                ]
            ).T
            prefactor_arr = gto_prefactor(l_2n_arr, self.radial_gaussian_width)
            gs *= prefactor_arr
            gto_overlap_slice = self.overlap_matrix[l_2n_arr, :][:, l_2n_arr]
            ortho_matrix = inverse_matrix_sqrt(gto_overlap_slice, self.rcond, self.tol).cpu().numpy()
            gs = np.einsum("jk,kl->jl", gs, ortho_matrix)
            all_gs = np.hstack((all_gs, gs))
        return all_gs[:, 1:]
