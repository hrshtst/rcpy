# ruff: noqa: ANN003, A005
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator, default_rng
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Callable

__SEED: int | None = None
__GLOBAL_RNG: Generator = default_rng()


def set_seed(seed: int | None) -> None:
    global __SEED  # noqa: PLW0603
    global __GLOBAL_RNG  # noqa: PLW0603
    if isinstance(seed, int | None):
        __SEED = seed
        __GLOBAL_RNG = default_rng(__SEED)
        np.random.seed(__SEED)  # noqa: NPY002
    else:
        msg = f"`seed` is expected to be an `int` or `None`, not {type(seed)}"
        raise TypeError(msg)


def get_seed() -> int | None:
    return __SEED


def get_rng(seed: int | Generator | None = None) -> Generator:
    if seed is None:
        return __GLOBAL_RNG
    if isinstance(seed, Generator):
        return seed
    if isinstance(seed, int):
        return default_rng(seed)
    msg = f"`seed` is expected to be an `int` or `None`, not {type(seed)}"
    raise TypeError(msg)


def _bernoulli_discrete_rvs(
    p: float = 0.5,
    value: float = 1.0,
    random_state: int | Generator | None = None,
) -> Callable:
    rng = get_rng(random_state)

    def rvs(size: int | tuple[int, ...] | None = None) -> np.ndarray:
        return rng.choice([value, -value], p=[p, 1 - p], replace=True, size=size)

    return rvs


def get_rvs(rng: int | Generator | None, dist: str, **kwargs) -> Callable:
    if dist == "custom_bernoulli":
        return _bernoulli_discrete_rvs(**kwargs, random_state=rng)
    if dist in dir(stats):
        distribution = getattr(stats, dist)
        return partial(distribution(**kwargs).rvs, random_state=rng)
    msg = f"`{dist}` is unavailable for probability distribution in 'scipy.stats'."
    raise ValueError(msg)


def noise(
    rng: Generator,
    dist: str = "normal",
    shape: int | tuple[int, ...] | None = None,
    gain: float = 1.0,
    **kwargs,
) -> np.ndarray:
    if abs(gain) > 0.0:
        return gain * getattr(rng, dist)(**kwargs, size=shape)
    if shape is None:
        shape = 1
    return np.zeros(shape, dtype=float)


# Local Variables:
# jinx-local-words: "bernoulli noqa scipy"
# End:
