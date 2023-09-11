from typing import Iterable, Literal, overload

import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse._base import _spbase

WeightsType = np.ndarray | csr_matrix | csc_matrix | coo_matrix


@overload
def initialize_weights(
    shape: tuple[int, ...],
    distribution=...,
    spectral_radius: float | None = ...,
    connectivity: float | None = ...,
    scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["dense"] = ...,
) -> np.ndarray:
    ...


@overload
def initialize_weights(
    shape: tuple[int, ...],
    distribution=...,
    spectral_radius: float | None = ...,
    connectivity: float | None = ...,
    scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["csr"] = ...,
) -> csr_matrix:
    ...


@overload
def initialize_weights(
    shape: tuple[int, ...],
    distribution=...,
    spectral_radius: float | None = ...,
    connectivity: float | None = ...,
    scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["csc"] = ...,
) -> csc_matrix:
    ...


@overload
def initialize_weights(
    shape: tuple[int, ...],
    distribution=...,
    spectral_radius: float | None = ...,
    connectivity: float | None = ...,
    scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["coo"] = ...,
) -> coo_matrix:
    ...


def initialize_weights(
    shape: tuple[int, ...],
    distribution=None,
    spectral_radius: float | None = None,
    connectivity: float | None = None,
    scaling: float | Iterable[float] | None = None,
    sparsity_type: Literal["dense", "csr", "csc", "coo"] = "csr",
) -> WeightsType | _spbase:
    if connectivity is None:
        connectivity = 0.1
    matrix = sparse.random(
        shape[0],
        shape[1],
        density=connectivity,
        format=sparsity_type,
        # random_state=rg,
        # data_rvs=rvs,
        dtype=float,
    )
    return matrix
