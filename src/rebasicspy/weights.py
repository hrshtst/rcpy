from typing import Iterable, Literal, overload

import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from rebasicspy._type import WeightsType, WeightsTypeVar
from rebasicspy.metrics import spectral_radius

_epsilon = 1e-8  # avoid division by zero when rescaling spectral radius


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
    sparsity_type: Literal["dense", "csr", "csc", "coo"] = "dense",
) -> WeightsType:
    if connectivity is None:
        connectivity = 0.1
    weights = sparse.random(
        shape[0],
        shape[1],
        density=connectivity,
        format=sparsity_type,
        # random_state=rg,
        # data_rvs=rvs,
        dtype=float,
    )
    if spectral_radius is not None:
        weights = _scale_spectral_radius(weights, spectral_radius)
    return weights


def _scale_spectral_radius(weights: WeightsTypeVar, sr: float) -> WeightsTypeVar:
    current_sr = spectral_radius(weights)
    if -_epsilon < current_sr < _epsilon:
        current_sr = _epsilon
    weights *= sr / current_sr
    return weights
