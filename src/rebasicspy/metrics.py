from typing import Iterable, Literal, overload

import numpy as np


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
