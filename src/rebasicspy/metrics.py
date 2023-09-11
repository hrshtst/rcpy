from typing import Iterable, Literal, overload

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import issparse

from rebasicspy._type import WeightsType


@overload
def mean_squared_error(
    y: Iterable[float | Iterable[float]],
    y_target: Iterable[float | Iterable[float]],
    root: bool = False,
    raw_values: Literal[False] = ...,
) -> float:
    ...


@overload
def mean_squared_error(
    y: Iterable[float | Iterable[float]],
    y_target: Iterable[float | Iterable[float]],
    root: bool = False,
    raw_values: Literal[True] = ...,
) -> np.ndarray:
    ...


def mean_squared_error(
    y: Iterable[float | Iterable[float]],
    y_target: Iterable[float | Iterable[float]],
    root: bool = False,
    raw_values: bool = False,
) -> float | np.ndarray:
    """Mean squared error."""

    y = np.array(y, dtype=float)
    y_target = np.array(y_target, dtype=float)
    errors = np.average((y - y_target) ** 2, axis=0)
    if root:
        errors = np.sqrt(errors)
    if raw_values:
        return errors
    return np.average(errors)


def spectral_radius(weights: WeightsType, maxiter: int | None = None) -> float:
    if issparse(weights):
        if maxiter is None:
            maxiter = weights.shape[0] * 20
        eigvals = spla.eigs(weights, k=1, which="LM", maxiter=maxiter, return_eigenvectors=False)
    else:
        eigvals = la.eigvals(weights)
    return np.max(np.abs(eigvals))
