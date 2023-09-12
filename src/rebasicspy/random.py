from functools import partial
from typing import Callable

import numpy as np
from numpy.random import Generator, default_rng
from scipy import stats

__SEED: int | None = None
__GLOBAL_RNG: Generator = default_rng()


def set_seed(seed: int | None):
    global __SEED
    global __GLOBAL_RNG
    if not isinstance(seed, int):
        pass
    __SEED = seed
    __GLOBAL_RNG = default_rng(__SEED)
    np.random.seed(__SEED)


def get_seed() -> int | None:
    return __SEED


def get_rng(seed: int | Generator | None = None) -> Generator:
    if seed is None:
        return __GLOBAL_RNG
    if isinstance(seed, Generator):
        return seed
    return default_rng(seed)


def _bernoulli_discrete_rvs(
    p: float = 0.5, value: float = 1.0, random_state: int | Generator | None = None
) -> Callable:
    rng = get_rng(random_state)

    def rvs(size: int | tuple[int, ...] | None = None):
        return rng.choice([value, -value], p=[p, 1 - p], replace=True, size=size)

    return rvs


def get_rvs(rng: int | Generator | None, dist: str, **kwargs) -> Callable:
    if dist == "custom_bernoulli":
        return _bernoulli_discrete_rvs(**kwargs, random_state=rng)
    elif dist in dir(stats):
        distribution = getattr(stats, dist)
        return partial(distribution(**kwargs).rvs, random_state=rng)
    else:
        raise ValueError(f"'{dist}' is unavailable for probability distribution in 'scipy.stats'.")


def noise(
    rng: Generator, dist: str = "normal", shape: int | tuple[int, ...] | None = None, gain: float = 1.0, **kwargs
) -> np.ndarray:
    if abs(gain) > 0.0:
        return gain * getattr(rng, dist)(**kwargs, size=shape)
    else:
        if shape is None:
            shape = 1
        return np.zeros(shape, dtype=float)
