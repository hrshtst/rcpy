import numpy as np
import pytest
from numpy.random import default_rng
from rebasicspy.random import get_rng, get_seed, noise, set_seed


def test_set_seed():
    assert get_seed() is None
    set_seed(12345)
    assert get_seed() == 12345


@pytest.mark.parametrize("seed", [None, 0, 123])
def test_get_rng_when_seed_is_none_return_global_rng(seed: int | None):
    set_seed(seed)
    rng1 = get_rng(None)
    rng2 = get_rng(None)
    assert rng1 is rng2


@pytest.mark.parametrize("seed", [0, 123])
def test_get_rng_when_seed_is_int_return_new_rng(seed: int):
    rng1 = get_rng(None)
    rng2 = get_rng(seed)
    assert rng1 is not rng2


def test_get_rng_when_seed_is_genrator_return_itself():
    seed = default_rng()
    rng = get_rng(seed)
    assert seed is rng


def test_noise_normal_distribution():
    rng = get_rng()
    arr1 = noise(rng, dist="normal", shape=10, loc=0.0, scale=1.0)
    arr2 = noise(rng, dist="normal", shape=10, loc=0.0, scale=1.0)
    assert not np.array_equal(arr1, arr2)
