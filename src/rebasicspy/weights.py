from typing import Callable, Iterable, Literal, overload

import numpy as np
from numpy.random import Generator
from scipy import sparse
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from rebasicspy._type import SparsityType, WeightsType
from rebasicspy.metrics import spectral_radius
from rebasicspy.random import get_rng, get_rvs

_epsilon = 1e-8  # avoid division by zero when rescaling spectral radius


@overload
def sparse_random(
    shape: tuple[int, ...],
    distribution: str = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["dense"] = ...,
    seed: int | Generator | None = ...,
    **kwargs,
) -> np.ndarray:
    ...


@overload
def sparse_random(
    shape: tuple[int, ...],
    distribution: str = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["csr"] = ...,
    seed: int | Generator | None = ...,
    **kwargs,
) -> csr_matrix:
    ...


@overload
def sparse_random(
    shape: tuple[int, ...],
    distribution: str = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["csc"] = ...,
    seed: int | Generator | None = ...,
    **kwargs,
) -> csc_matrix:
    ...


@overload
def sparse_random(
    shape: tuple[int, ...],
    distribution: str = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["coo"] = ...,
    seed: int | Generator | None = ...,
    **kwargs,
) -> coo_matrix:
    ...


def sparse_random(
    shape: tuple[int, ...],
    distribution: str = "uniform",
    connectivity: float | None = None,
    sparsity_type: SparsityType = "dense",
    seed: int | Generator | None = None,
    **kwargs,
) -> WeightsType:
    if connectivity is None:
        connectivity = 0.1
    elif connectivity < 0 or connectivity > 1:
        raise ValueError(f"`connectivity` expected to be 0 <= connectivity <= 1.")

    rng = get_rng(seed)
    rvs = get_rvs(rng, dist=distribution, **kwargs)
    weights = sparse.random(
        shape[0],
        shape[1],
        density=connectivity,
        format=sparsity_type,
        random_state=rng,
        data_rvs=rvs,
        dtype=float,
    )
    return weights


@overload
def uniform(
    shape: tuple[int, ...],
    low: float = ...,
    high: float = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["dense"] = ...,
    seed: int | Generator | None = ...,
) -> np.ndarray:
    ...


@overload
def uniform(
    shape: tuple[int, ...],
    low: float = ...,
    high: float = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["csr", "csc", "coo"] = ...,
    seed: int | Generator | None = ...,
) -> csr_matrix | csc_matrix | coo_matrix:
    ...


def uniform(
    shape: tuple[int, ...],
    low: float = -1.0,
    high: float = 1.0,
    connectivity: float | None = None,
    sparsity_type: SparsityType = "dense",
    seed: int | Generator | None = None,
) -> WeightsType:
    if low > high:
        raise ValueError(f"`high` boundary expected to be bigger than `low` boundary.")
    return sparse_random(
        shape,
        distribution="uniform",
        connectivity=connectivity,
        sparsity_type=sparsity_type,
        seed=seed,
        loc=low,
        scale=high - low,
    )


@overload
def normal(
    shape: tuple[int, ...],
    loc: float = ...,
    scale: float = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["dense"] = ...,
    seed: int | Generator | None = ...,
) -> np.ndarray:
    ...


@overload
def normal(
    shape: tuple[int, ...],
    loc: float = ...,
    scale: float = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["csr", "csc", "coo"] = ...,
    seed: int | Generator | None = ...,
) -> csr_matrix | csc_matrix | coo_matrix:
    ...


def normal(
    shape: tuple[int, ...],
    loc: float = 0.0,
    scale: float = 1.0,
    connectivity: float | None = None,
    sparsity_type: SparsityType = "dense",
    seed: int | Generator | None = None,
) -> WeightsType:
    return sparse_random(
        shape,
        distribution="norm",
        connectivity=connectivity,
        sparsity_type=sparsity_type,
        seed=seed,
        loc=loc,
        scale=scale,
    )


@overload
def bernoulli(
    shape: tuple[int, ...],
    p: float = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["dense"] = ...,
    seed: int | Generator | None = ...,
) -> np.ndarray:
    ...


@overload
def bernoulli(
    shape: tuple[int, ...],
    p: float = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["csr", "csc", "coo"] = ...,
    seed: int | Generator | None = ...,
) -> csr_matrix | csc_matrix | coo_matrix:
    ...


def bernoulli(
    shape: tuple[int, ...],
    p: float = 0.5,
    connectivity: float | None = None,
    sparsity_type: SparsityType = "dense",
    seed: int | Generator | None = None,
) -> WeightsType:
    if 1 < p or p < 0:
        raise ValueError("'p' must be <= 1 and >= 0.")
    return sparse_random(
        shape,
        distribution="custom_bernoulli",
        connectivity=connectivity,
        sparsity_type=sparsity_type,
        seed=seed,
        p=p,
    )


def _scale_spectral_radius(weights: WeightsType, sr: float) -> WeightsType:
    current_sr = spectral_radius(weights)
    if -_epsilon < current_sr < _epsilon:
        current_sr = _epsilon
    weights *= sr / current_sr
    return weights


@overload
def initialize_weights(
    shape: tuple[int, ...],
    w_initializer: Callable[..., WeightsType],
    spectral_radius: float | None = ...,
    input_scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["dense"] = ...,
    **kwargs,
) -> np.ndarray:
    ...


@overload
def initialize_weights(
    shape: tuple[int, ...],
    w_initializer: Callable[..., WeightsType],
    spectral_radius: float | None = ...,
    input_scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["csr"] = ...,
    **kwargs,
) -> csr_matrix:
    ...


@overload
def initialize_weights(
    shape: tuple[int, ...],
    w_initializer: Callable[..., WeightsType],
    spectral_radius: float | None = ...,
    input_scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["csc"] = ...,
    **kwargs,
) -> csc_matrix:
    ...


@overload
def initialize_weights(
    shape: tuple[int, ...],
    w_initializer: Callable[..., WeightsType],
    spectral_radius: float | None = ...,
    input_scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["coo"] = ...,
    **kwargs,
) -> coo_matrix:
    ...


def initialize_weights(
    shape: tuple[int, ...],
    w_initializer: Callable[..., WeightsType],
    spectral_radius: float | None = None,
    input_scaling: float | Iterable[float] | None = None,
    sparsity_type: SparsityType = "dense",
    **kwargs,
) -> WeightsType:
    w = w_initializer(shape, sparsity_type=sparsity_type, **kwargs)
    if spectral_radius is not None:
        w = _scale_spectral_radius(w, spectral_radius)
    # if input_scaling is not None:
    #     w = _scale_input_scaling(w, input_scaling)
    return w
