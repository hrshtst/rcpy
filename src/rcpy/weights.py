# ruff: noqa: ANN003
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import ArpackNoConvergence

from rcpy.metrics import spectral_radius
from rcpy.random import get_rng, get_rvs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numpy.random import Generator

    from rcpy._type import SparsityType, WeightsType

_epsilon = 1e-8  # avoid division by zero when rescaling spectral radius


@overload
def sparse_random(
    shape: tuple[int, ...],
    distribution: str = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["dense"] = ...,
    seed: int | Generator | None = ...,
    **kwargs,
) -> np.ndarray: ...


@overload
def sparse_random(
    shape: tuple[int, ...],
    distribution: str = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["csr"] = ...,
    seed: int | Generator | None = ...,
    **kwargs,
) -> csr_matrix: ...


@overload
def sparse_random(
    shape: tuple[int, ...],
    distribution: str = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["csc"] = ...,
    seed: int | Generator | None = ...,
    **kwargs,
) -> csc_matrix: ...


@overload
def sparse_random(
    shape: tuple[int, ...],
    distribution: str = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["coo"] = ...,
    seed: int | Generator | None = ...,
    **kwargs,
) -> coo_matrix: ...


def sparse_random(
    shape: tuple[int, ...],
    distribution: str = "uniform",
    connectivity: float | None = None,
    sparsity_type: SparsityType = "dense",
    seed: int | Generator | None = None,
    **kwargs,
) -> WeightsType:
    if connectivity is None:
        connectivity = 1.0
    elif connectivity < 0 or connectivity > 1:
        msg = "`connectivity` expected to be 0 <= connectivity <= 1."
        raise ValueError(msg)

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

    if isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    return weights


@overload
def uniform(
    shape: tuple[int, ...],
    low: float = ...,
    high: float = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["dense"] = ...,
    seed: int | Generator | None = ...,
) -> np.ndarray: ...


@overload
def uniform(
    shape: tuple[int, ...],
    low: float = ...,
    high: float = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["csr", "csc", "coo"] = ...,
    seed: int | Generator | None = ...,
) -> csr_matrix | csc_matrix | coo_matrix: ...


def uniform(
    shape: tuple[int, ...],
    low: float = -1.0,
    high: float = 1.0,
    connectivity: float | None = None,
    sparsity_type: SparsityType = "dense",
    seed: int | Generator | None = None,
) -> WeightsType:
    if low > high:
        msg = "`high` boundary expected to be bigger than `low` boundary."
        raise ValueError(msg)
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
) -> np.ndarray: ...


@overload
def normal(
    shape: tuple[int, ...],
    loc: float = ...,
    scale: float = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["csr", "csc", "coo"] = ...,
    seed: int | Generator | None = ...,
) -> csr_matrix | csc_matrix | coo_matrix: ...


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
) -> np.ndarray: ...


@overload
def bernoulli(
    shape: tuple[int, ...],
    p: float = ...,
    connectivity: float | None = ...,
    sparsity_type: Literal["csr", "csc", "coo"] = ...,
    seed: int | Generator | None = ...,
) -> csr_matrix | csc_matrix | coo_matrix: ...


def bernoulli(
    shape: tuple[int, ...],
    p: float = 0.5,
    connectivity: float | None = None,
    sparsity_type: SparsityType = "dense",
    seed: int | Generator | None = None,
) -> WeightsType:
    if p > 1 or p < 0:
        msg = "'p' must be <= 1 and >= 0."
        raise ValueError(msg)
    return sparse_random(
        shape,
        distribution="custom_bernoulli",
        connectivity=connectivity,
        sparsity_type=sparsity_type,
        seed=seed,
        p=p,
    )


def ones(shape: int | tuple[int, ...], **kwargs) -> np.ndarray:
    _ = kwargs
    return np.ones(shape, dtype=float)


def zeros(shape: int | tuple[int, ...], **kwargs) -> np.ndarray:
    _ = kwargs
    return np.zeros(shape, dtype=float)


@overload
def initialize_weights(
    shape: tuple[int, ...],
    w_initializer: Callable[..., WeightsType],
    spectral_radius: float | None = ...,
    scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["dense"] = ...,
    **kwargs,
) -> np.ndarray: ...


@overload
def initialize_weights(
    shape: tuple[int, ...],
    w_initializer: Callable[..., WeightsType],
    spectral_radius: float | None = ...,
    scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["csr"] = ...,
    **kwargs,
) -> csr_matrix: ...


@overload
def initialize_weights(
    shape: tuple[int, ...],
    w_initializer: Callable[..., WeightsType],
    spectral_radius: float | None = ...,
    scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["csc"] = ...,
    **kwargs,
) -> csc_matrix: ...


@overload
def initialize_weights(
    shape: tuple[int, ...],
    w_initializer: Callable[..., WeightsType],
    spectral_radius: float | None = ...,
    scaling: float | Iterable[float] | None = ...,
    sparsity_type: Literal["coo"] = ...,
    **kwargs,
) -> coo_matrix: ...


def initialize_weights(
    shape: tuple[int, ...],
    w_initializer: Callable[..., WeightsType],
    spectral_radius: float | None = None,
    scaling: float | Iterable[float] | None = None,
    sparsity_type: SparsityType = "dense",
    **kwargs,
) -> WeightsType:
    iteration = 10
    while iteration > 0:
        w = w_initializer(shape, sparsity_type=sparsity_type, **kwargs)
        try:
            if spectral_radius is not None:
                w = _scale_spectral_radius(w, spectral_radius)
            if scaling is not None:
                w = _scale_inputs(w, scaling)
        except ArpackNoConvergence:
            iteration -= 1
            warnings.warn("Re-sampling initial weights", stacklevel=2)
        else:
            return w
    msg = "No convergence: did not find any eigenvalues to sufficient accuracy."
    raise RuntimeError(msg)


def _scale_spectral_radius(weights: WeightsType, sr: float) -> WeightsType:
    current_sr = spectral_radius(weights)
    if -_epsilon < current_sr < _epsilon:
        current_sr = _epsilon
    weights *= sr / current_sr
    return weights


def _scale_inputs(weights: WeightsType, scaling: float | Iterable[float]) -> WeightsType:
    if isinstance(scaling, float | int):
        return weights * scaling
    if weights.shape is not None and len(list(scaling)) == weights.shape[1]:
        return weights * diags(scaling)
    msg = "The size of `scaling` is mismatched with `weights`."
    raise ValueError(msg)


# Local Variables:
# jinx-local-words: "bernoulli csc csr noqa rescaling"
# End:
