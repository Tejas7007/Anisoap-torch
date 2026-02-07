import sys
import warnings
from itertools import product

import metatensor
import numpy as np
import torch
from anisoap_rust_lib import compute_moments, compute_moments_batch
from featomic import NeighborList
from metatensor import (
    Labels,
    TensorBlock,
    TensorMap,
)
from scipy.spatial.transform import Rotation
from skmatter.preprocessing import StandardFlexibleScaler
from tqdm.auto import tqdm

from anisoap.representations.radial_basis import (
    GTORadialBasis,
    MonomialBasis,
)
from anisoap.utils.metatensor_utils import (
    ClebschGordanReal,
    cg_combine,
    standardize_keys,
)
from anisoap.utils.moment_generator import *
from anisoap.utils.spherical_to_cartesian import spherical_to_cartesian


def pairwise_ellip_expansion(
    lmax,
    neighbor_list,
    types,
    frame_to_global_atom_idx,
    rotation_matrices,
    ellipsoid_lengths,
    sph_to_cart,
    radial_basis,
    show_progress=False,
    rust_moments=True,
    device=None,
):
    r"""Computes pairwise expansion using PyTorch for all array operations."""
    dev = torch.device(device or "cpu")
    tensorblock_list = []
    keys = np.asarray(neighbor_list.keys, dtype=int)
    keys = [tuple(i) + (l,) for i in keys for l in range(lmax + 1)]
    num_ns = radial_basis.get_num_radial_functions()
    maxdeg = np.max(np.arange(lmax + 1) + 2 * np.array(num_ns))

    # Precompute scaled spherical-to-Cartesian transforms as torch tensors
    solid_harm_prefact = np.sqrt((4 * np.pi) / (np.arange(lmax + 1) * 2 + 1))
    scaled_sph_to_cart = []
    for l in range(lmax + 1):
        t = torch.from_numpy(sph_to_cart[l] / solid_harm_prefact[l]).to(dtype=torch.float64, device=dev)
        scaled_sph_to_cart.append(t)

    # Convert global arrays to torch
    rot_matrices_t = torch.as_tensor(rotation_matrices, dtype=torch.float64, device=dev)
    ellip_lengths_t = torch.as_tensor(ellipsoid_lengths, dtype=torch.float64, device=dev)
    ftga = torch.as_tensor(frame_to_global_atom_idx, dtype=torch.long, device=dev)

    for center_types in types:
        for neighbor_types in types:
            if (center_types, neighbor_types) in neighbor_list.keys:
                nl_block = neighbor_list.block(
                    first_atom_type=center_types,
                    second_atom_type=neighbor_types,
                )

                n_samples = nl_block.values.shape[0]

                if n_samples == 0:
                    for l in range(lmax + 1):
                        block = TensorBlock(
                            values=np.zeros((0, 2 * l + 1, num_ns[l])),
                            samples=nl_block.samples,
                            components=[
                                Labels(
                                    ["spherical_component_m"],
                                    np.asarray(list(range(-l, l + 1)), np.int32).reshape(-1, 1),
                                )
                            ],
                            properties=Labels(
                                ["n"],
                                np.asarray(list(range(num_ns[l])), np.int32).reshape(-1, 1),
                            ),
                        )
                        tensorblock_list.append(block)
                    continue

                # === BATCH EXTRACT all pair data at once ===
                samples_arr = np.array(
                    [(s["system"], s["first_atom"], s["second_atom"]) for s in nl_block.samples]
                )
                frame_idxs = torch.as_tensor(samples_arr[:, 0], dtype=torch.long, device=dev)
                j_atoms = torch.as_tensor(samples_arr[:, 2], dtype=torch.long, device=dev)
                j_globals = ftga[frame_idxs] + j_atoms

                # Extract all r_ij vectors: (N, 3)
                all_r_ij = np.asarray(nl_block.values[:, :, 0])

                # Gather all rotations and lengths via torch indexing
                all_rot = rot_matrices_t[j_globals]       # (N, 3, 3) torch
                all_lengths = ellip_lengths_t[j_globals]   # (N, 3) torch

                # === BATCH compute gaussian parameters (torch internally) ===
                all_precision, all_center, all_constant = radial_basis.compute_gaussian_parameters_batch(
                    all_r_ij, all_lengths.cpu().numpy(), all_rot.cpu().numpy()
                )

                # Precompute scaling using torch
                all_lengths_np = all_lengths.cpu().numpy()
                all_length_norm = (np.prod(all_lengths_np, axis=1) * (2.0 * np.pi) ** 1.5) ** -1.0
                all_scale = np.exp(-0.5 * all_constant) * all_length_norm

                # === BATCH compute moments (Rust FFI — needs numpy) ===
                D = maxdeg + 1
                if rust_moments:
                    mats_flat = all_precision.reshape(n_samples, 9)
                    moments_flat = np.array(compute_moments_batch(mats_flat, all_center, maxdeg))
                    all_moments = moments_flat.reshape(n_samples, D, D, D)
                else:
                    all_moments = np.empty((n_samples, D, D, D))
                    for idx in range(n_samples):
                        all_moments[idx] = compute_moments_inefficient_implementation(
                            all_precision[idx], all_center[idx], maxdeg=maxdeg
                        )

                # Scale and convert to torch
                all_moments *= all_scale[:, np.newaxis, np.newaxis, np.newaxis]
                all_moments_t = torch.from_numpy(all_moments).to(dtype=torch.float64, device=dev)

                # === BATCH einsum per angular channel (torch) ===
                for l in range(lmax + 1):
                    deg = l + 2 * (num_ns[l] - 1)
                    moments_l = all_moments_t[:, : deg + 1, : deg + 1, : deg + 1]

                    values_l = torch.einsum(
                        "mnpqr, bpqr->bmn",
                        scaled_sph_to_cart[l],
                        moments_l,
                    ).cpu().numpy()

                    block = TensorBlock(
                        values=values_l,
                        samples=nl_block.samples,
                        components=[
                            Labels(
                                ["spherical_component_m"],
                                np.asarray(list(range(-l, l + 1)), np.int32).reshape(-1, 1),
                            )
                        ],
                        properties=Labels(
                            ["n"],
                            np.asarray(list(range(num_ns[l])), np.int32).reshape(-1, 1),
                        ),
                    )
                    tensorblock_list.append(block)

    pairwise_ellip_feat = TensorMap(
        Labels(
            ["types_center", "types_neighbor", "angular_channel"],
            np.asarray(keys, dtype=np.int32),
        ),
        tensorblock_list,
    )
    return pairwise_ellip_feat


def contract_pairwise_feat(pair_ellip_feat, types, show_progress=False, device=None):
    """Vectorized contraction using PyTorch scatter operations."""
    dev = torch.device(device or "cpu")

    ellip_keys = list(
        set([tuple(list(x)[:1] + list(x)[2:]) for x in pair_ellip_feat.keys])
    )
    ellip_keys.sort()
    ellip_blocks = []
    property_names = pair_ellip_feat.property_names + ["neighbor_types"]

    for key in ellip_keys:
        contract_blocks = []
        contract_properties = []
        contract_samples = []

        for ele in types:
            selection = Labels(names=["types_neighbor"], values=np.array([[ele]]))
            blockidx = pair_ellip_feat.blocks_matching(selection=selection)
            sel_blocks = [
                pair_ellip_feat.block(i)
                for i in blockidx
                if key == tuple(
                    list(pair_ellip_feat.keys[i])[:1]
                    + list(pair_ellip_feat.keys[i])[2:]
                )
            ]

            if not len(sel_blocks):
                continue
            assert len(sel_blocks) == 1
            block = sel_blocks[0]

            # Vectorized group-by-sum using torch scatter_add
            systems = np.array(block.samples["system"])
            first_atoms = np.array(block.samples["first_atom"])

            pair_keys = np.column_stack([systems, first_atoms])
            unique_pairs, inverse_idx = np.unique(pair_keys, axis=0, return_inverse=True)

            vals_t = torch.from_numpy(np.array(block.values)).to(dtype=torch.float64, device=dev)
            inv_t = torch.from_numpy(inverse_idx).to(dtype=torch.long, device=dev)

            # scatter_add: (n_unique, components, properties)
            summed = torch.zeros(
                len(unique_pairs), *vals_t.shape[1:],
                dtype=torch.float64, device=dev,
            )
            # Expand inverse index to match value dimensions
            inv_expanded = inv_t.unsqueeze(-1).unsqueeze(-1).expand_as(vals_t)
            summed.scatter_add_(0, inv_expanded, vals_t)

            block_samples = [tuple(p) for p in unique_pairs]
            block_values = list(summed.cpu().numpy())

            contract_blocks.append(block_values)
            contract_samples.append(block_samples)
            contract_properties.append([tuple(p) + (ele,) for p in block.properties])

        all_block_samples = sorted(list(set().union(*contract_samples)))

        all_block_values = np.zeros(
            (len(all_block_samples),)
            + block.values.shape[1:]
            + (len(contract_blocks),)
        )

        sample_to_idx = {s: i for i, s in enumerate(all_block_samples)}

        for iele, (elem_samples, elem_values) in enumerate(
            zip(contract_samples, contract_blocks)
        ):
            idx_arr = np.array([sample_to_idx[s] for s in elem_samples])
            all_block_values[idx_arr, :, :, iele] = np.array(elem_values)

        new_block = TensorBlock(
            values=all_block_values.reshape(
                all_block_values.shape[0], all_block_values.shape[1], -1
            ),
            samples=Labels(["type", "center"], np.asarray(all_block_samples, np.int32)),
            components=block.components,
            properties=Labels(
                list(property_names),
                np.asarray(np.vstack(contract_properties), np.int32),
            ),
        )
        ellip_blocks.append(new_block)

    ellip = TensorMap(
        Labels(
            ["types_center", "angular_channel"],
            np.asarray(ellip_keys, dtype=np.int32),
        ),
        ellip_blocks,
    )
    return ellip


class EllipsoidalDensityProjection:
    """Class for computing spherical projection coefficients with PyTorch backend.

    Parameters
    ----------
    max_angular : int
    radial_basis_name : str
    cutoff_radius : float
    compute_gradients : bool
    subtract_center_contribution : bool
    radial_gaussian_width : float, optional
    max_radial : None, int, list of int
    rotation_key : string
    rotation_type : string
    device : str or torch.device, optional
        PyTorch device ('cpu', 'cuda', 'mps'). Defaults to 'cpu'.
    """

    def __init__(
        self,
        max_angular,
        radial_basis_name,
        cutoff_radius,
        compute_gradients=False,
        subtract_center_contribution=False,
        radial_gaussian_width=None,
        max_radial=None,
        rotation_key="quaternion",
        rotation_type="quaternion",
        basis_rcond=0,
        basis_tol=1e-8,
        device=None,
    ):
        self.max_angular = max_angular
        self.cutoff_radius = cutoff_radius
        self.compute_gradients = compute_gradients
        self.subtract_center_contribution = subtract_center_contribution
        self.radial_basis_name = radial_basis_name
        self.device = torch.device(device or "cpu")

        if compute_gradients:
            raise NotImplementedError("Sorry! Gradients have not yet been implemented")

        radial_hypers = {}
        radial_hypers["radial_gaussian_width"] = radial_gaussian_width
        radial_hypers["max_angular"] = max_angular
        radial_hypers["cutoff_radius"] = cutoff_radius
        radial_hypers["max_radial"] = max_radial
        radial_hypers["rcond"] = basis_rcond
        radial_hypers["tol"] = basis_tol
        radial_hypers["device"] = self.device

        if type(cutoff_radius) == int:
            raise ValueError(
                "r_cut is set as an integer, which could cause overflow errors. Pass in float"
            )
        if radial_basis_name == "gto":
            if radial_hypers.get("radial_gaussian_width") is None:
                raise ValueError("Gaussian width must be provided with GTO basis")
            if type(radial_hypers.get("radial_gaussian_width")) == int:
                raise ValueError(
                    "radial_gaussian_width is set as an integer, which could cause overflow errors. Pass in float."
                )
            self.radial_basis = GTORadialBasis(**radial_hypers)
        elif radial_basis_name == "monomial":
            rgw = radial_hypers.pop("radial_gaussian_width")
            if rgw is not None:
                raise ValueError("Gaussian width can only be provided with GTO basis")
            self.radial_basis = MonomialBasis(**radial_hypers)
        else:
            raise NotImplementedError(
                f"{self.radial_basis_name} is not an implemented basis"
                ". Try 'monomial' or 'gto'"
            )

        self.num_ns = self.radial_basis.get_num_radial_functions()
        self.sph_to_cart = spherical_to_cartesian(self.max_angular, self.num_ns)

        if rotation_type not in ["quaternion", "matrix"]:
            raise ValueError(
                "We have only implemented transforming quaternions (`quaternion`) and rotation matrices (`matrix`)."
            )
        elif rotation_type == "quaternion":
            self.rotation_maker = lambda q: Rotation.from_quat([*q[1:], q[0]])
            warnings.warn(
                "In quaternion mode, quaternions are assumed to be in (w,x,y,z) format."
            )
        else:
            self.rotation_maker = Rotation.from_matrix

        self.rotation_key = rotation_key

    def transform(self, frames, show_progress=False, normalize=True, rust_moments=True):
        """Computes features for frames using PyTorch backend.

        Parameters
        ----------
        frames : list of ase.Atoms
        show_progress : bool
        normalize : bool
        rust_moments : bool

        Returns
        -------
        TensorMap
        """
        self.frames = frames
        dev = self.device

        num_frames = len(frames)
        types = set()
        self.num_atoms_per_frame = np.zeros((num_frames), int)

        for i, f in enumerate(self.frames):
            self.num_atoms_per_frame[i] = len(f)
            for atom in f:
                types.add(atom.number)

        self.num_atoms_total = sum(self.num_atoms_per_frame)
        types = sorted(types)

        self.num_atoms_per_frame = np.array([len(frame) for frame in frames])
        self.feature_gradients = 0

        frame_generator = tqdm(
            self.frames, disable=(not show_progress), desc="Computing neighborlist"
        )

        self.frame_to_global_atom_idx = np.zeros((num_frames), int)
        for n in range(1, num_frames):
            self.frame_to_global_atom_idx[n] = (
                self.num_atoms_per_frame[n - 1] + self.frame_to_global_atom_idx[n - 1]
            )

        # Build rotation matrices and ellipsoid lengths using torch
        rotation_matrices = torch.zeros(self.num_atoms_total, 3, 3, dtype=torch.float64, device=dev)
        ellipsoid_lengths = torch.zeros(self.num_atoms_total, 3, dtype=torch.float64, device=dev)

        for i in range(num_frames):
            for j in range(self.num_atoms_per_frame[i]):
                j_global = self.frame_to_global_atom_idx[i] + j
                if self.rotation_key in frames[i].arrays:
                    rot_np = self.rotation_maker(
                        frames[i].arrays[self.rotation_key][j]
                    ).as_matrix()
                    rotation_matrices[j_global] = torch.from_numpy(rot_np).to(dtype=torch.float64, device=dev)
                else:
                    warnings.warn(
                        f"Frame {i} does not have rotations stored, this may cause errors down the line."
                    )

                ellipsoid_lengths[j_global] = torch.tensor([
                    frames[i].arrays["c_diameter[1]"][j] / 2,
                    frames[i].arrays["c_diameter[2]"][j] / 2,
                    frames[i].arrays["c_diameter[3]"][j] / 2,
                ], dtype=torch.float64, device=dev)

        self.nl = NeighborList(
            cutoff=self.cutoff_radius,
            full_neighbor_list=True,
            self_pairs=(not self.subtract_center_contribution),
        ).compute(frame_generator)

        pairwise_ellip_feat = pairwise_ellip_expansion(
            self.max_angular,
            self.nl,
            types,
            self.frame_to_global_atom_idx,
            rotation_matrices.cpu().numpy(),
            ellipsoid_lengths.cpu().numpy(),
            self.sph_to_cart,
            self.radial_basis,
            show_progress,
            rust_moments=rust_moments,
            device=dev,
        )

        features = contract_pairwise_feat(pairwise_ellip_feat, types, show_progress, device=dev)
        if normalize:
            normalized_features = self.radial_basis.orthonormalize_basis(features)
            return normalized_features
        else:
            return features

    def power_spectrum(
        self, frames, mean_over_samples=True, show_progress=False, rust_moments=True
    ):
        """Helper function to compute the power spectrum of AniSOAP."""
        mycg = ClebschGordanReal(self.max_angular)

        if frames[0].arrays is None:
            raise ValueError("frames cannot be none")
        required_attributes = [
            "c_diameter[1]", "c_diameter[2]", "c_diameter[3]",
            "c_q", "positions", "numbers",
        ]

        for index, frame in enumerate(frames):
            array = frame.arrays
            for attr in required_attributes:
                if attr not in array:
                    raise ValueError(
                        f"frame at index {index} is missing a required attribute '{attr}'"
                    )
                if "quaternion" in array:
                    raise ValueError(f"frame should contain c_q rather than quaternion")

        mvg_coeffs = self.transform(
            frames, show_progress=show_progress, rust_moments=rust_moments
        )
        mvg_nu1 = standardize_keys(mvg_coeffs)

        mvg_nu2 = cg_combine(
            mvg_nu1,
            mvg_nu1,
            clebsch_gordan=mycg,
            lcut=0,
            other_keys_match=["types_center"],
        )

        if mean_over_samples:
            x_asoap_raw = metatensor.mean_over_samples(mvg_nu2, sample_names="center")
            x_asoap_raw = np.hstack(
                [block.values.squeeze() for block in x_asoap_raw.blocks()]
            )
            return x_asoap_raw
        else:
            return mvg_nu2
