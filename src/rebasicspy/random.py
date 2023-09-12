import numpy as np
from numpy.random import Generator, default_rng

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


def noise(
    rng: Generator, dist: str = "normal", shape: int | tuple[int, ...] | None = None, gain: float = 1.0, **kwargs
) -> np.ndarray:
    if abs(gain) > 0.0:
        return gain * getattr(rng, dist)(**kwargs, size=shape)
    else:
        if shape is None:
            shape = 1
        return np.zeros(shape, dtype=float)
