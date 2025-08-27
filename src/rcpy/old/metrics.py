from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import issparse

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rcpy._type import WeightsType


@overload
def mean_squared_error(
    y: Iterable[float | Iterable[float]],
    y_target: Iterable[float | Iterable[float]],
    *,
    root: bool = False,
    raw_values: Literal[False] = ...,
) -> float: ...


@overload
def mean_squared_error(
    y: Iterable[float | Iterable[float]],
    y_target: Iterable[float | Iterable[float]],
    *,
    root: bool = False,
    raw_values: Literal[True],
) -> np.ndarray: ...


@overload
def mean_squared_error(
    y: Iterable[float | Iterable[float]],
    y_target: Iterable[float | Iterable[float]],
    *,
    root: bool = ...,
    raw_values: bool = ...,
) -> float | np.ndarray: ...


def mean_squared_error(
    y: Iterable[float | Iterable[float]],
    y_target: Iterable[float | Iterable[float]],
    *,
    root: bool = False,
    raw_values: bool = False,
) -> float | np.ndarray:
    """Mean squared error."""
    y = np.asarray(y, dtype=float)
    y_target = np.asarray(y_target, dtype=float)
    errors = np.average((y - y_target) ** 2, axis=0)
    if root:
        errors = np.sqrt(errors)
    if raw_values:
        return errors
    return np.average(errors)


def spectral_radius(weights: WeightsType, maxiter: int | None = None) -> float:
    if issparse(weights):
        if maxiter is None and weights.shape is not None:
            maxiter = weights.shape[0] * 20
        eigvals = spla.eigs(weights, k=1, which="LM", maxiter=maxiter, return_eigenvectors=False)
    else:
        eigvals = la.eigvals(weights)
    return np.max(np.abs(eigvals))
