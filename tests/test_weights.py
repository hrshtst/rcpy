import warnings
from typing import Literal

import numpy as np
import pytest
from rebasicspy.weights import bernoulli, initialize_weights, normal, ones, sparse_random, uniform, zeros
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix


@pytest.mark.parametrize(
    "shape,sparsity_type", [((10, 10), "dense"), ((10, 1), "csr"), ((1, 10), "csc"), ((20, 20), "coo")]
)
def test_sparse_random_return_specified_shape(shape: tuple[int, int], sparsity_type):
    w = sparse_random(shape, sparsity_type=sparsity_type)
    assert w.shape == shape


@pytest.mark.parametrize(
    "sparsity_type,expected_type",
    [("dense", np.ndarray), ("csr", csr_matrix), ("csc", csc_matrix), ("coo", coo_matrix)],
)
def test_sparse_random_specify_sparsity_type(sparsity_type, expected_type):
    w = sparse_random((5, 5), sparsity_type=sparsity_type)
    assert isinstance(w, expected_type)


@pytest.mark.parametrize(
    "sparsity_type,shape,connectivity", [("coo", (10, 10), 0.02), ("csr", (10, 10), 0.05), ("csc", (10, 10), 0.15)]
)
def test_sparse_random_specify_connectivity(sparsity_type, shape: tuple[int, int], connectivity: float):
    w = sparse_random(shape, sparsity_type=sparsity_type, connectivity=connectivity)
    actual = w.nnz / w.toarray().size
    assert actual == connectivity


@pytest.mark.parametrize("shape,connectivity", [((10, 10), 0.02), ((10, 10), 0.05), ((10, 10), 0.15)])
def test_sparse_random_specify_connectivity_with_dense_array(shape: tuple[int, int], connectivity: float):
    w = sparse_random(shape, sparsity_type="dense", connectivity=connectivity)
    actual = np.count_nonzero(w) / w.size
    assert actual == connectivity


@pytest.mark.parametrize("sparsity_type", ["dense", "csr", "csc", "coo"])
def test_sparse_random_if_connectivity_is_none_then_it_should_be_default_value(sparsity_type):
    default = 0.1
    w = sparse_random((10, 10), sparsity_type=sparsity_type, connectivity=None)
    if isinstance(w, np.ndarray):
        actual = np.count_nonzero(w) / w.size
    else:
        actual = w.nnz / w.toarray().size
    assert actual == default


@pytest.mark.parametrize("connectivity", [-0.1, 1.1])
def test_sparse_random_connectivity_must_be_between_zero_and_one(connectivity: float):
    with pytest.raises(ValueError):
        _ = sparse_random((10, 10), connectivity=connectivity)


def test_sparse_random_no_warning_raised_when_very_small_connectivity():
    connectivity = 1e-8
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = sparse_random((100, 100), connectivity=connectivity, sparsity_type="coo")


@pytest.mark.parametrize("low,high", [(-1.0, 1.0), (-0.1, 0.1), (0.0, 1.0)])
def test_uniform_specifiy_low_and_high(low: float, high: float):
    w = uniform((3, 3), low=low, high=high, connectivity=1.0)
    for i in np.ravel(w):
        assert low < i < high


def test_uniform_raise_exception_when_low_is_bigger_than_high():
    with pytest.raises(ValueError):
        _ = uniform((3, 3), low=1, high=-1, connectivity=1.0)


@pytest.mark.parametrize("loc,scale", [(0.0, 1.0), (0.0, 2.0)])
def test_normal_specifiy_loc_and_scale(loc: float, scale: float):
    w = normal((20, 20), loc=loc, scale=scale, connectivity=1.0)
    hist = np.histogram(w, bins=9)
    assert np.max(hist[0]) == hist[0][4]


@pytest.mark.parametrize("p", [0.5, 0.3, 0.8])
def test_bernoulli_specifiy_p(p: float):
    w = bernoulli((10, 10), p=p, connectivity=1.0)
    for i in np.ravel(w):
        assert i == 1.0 or i == -1.0


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_bernoulli_raise_exception_when_unexpected_p_given(p: float):
    with pytest.raises(ValueError):
        _ = bernoulli((10, 10), p=p, connectivity=1.0)


def test_ones():
    arr = ones(5)
    assert arr.shape == (5,)
    assert arr.dtype == float
    for i in arr:
        assert i == 1.0


def test_zeros():
    arr = zeros(5)
    assert arr.shape == (5,)
    assert arr.dtype == float
    for i in arr:
        assert i == 0.0


@pytest.mark.parametrize("sr", [0.95, 1.0, 1.3])
@pytest.mark.parametrize(
    "sparsity_type,expected_type",
    [("csr", csr_matrix), ("csc", csc_matrix), ("coo", coo_matrix)],
)
def test_initialize_weights_scale_spectral_radius(
    sr: float, sparsity_type: Literal["csr", "csc", "coo"], expected_type
):
    w = initialize_weights((100, 100), uniform, spectral_radius=sr, sparsity_type=sparsity_type)
    arr = w.toarray()
    actual = max(abs(np.linalg.eigvals(arr)))
    assert isinstance(w, expected_type)
    assert actual == pytest.approx(sr)


@pytest.mark.parametrize("sr", [0.95, 1.0, 1.3])
def test_initialize_weights_scale_spectral_radius_with_dense_array(sr: float):
    w = initialize_weights((100, 100), uniform, spectral_radius=sr, sparsity_type="dense")
    actual = max(abs(np.linalg.eigvals(w)))
    assert isinstance(w, np.ndarray)
    assert actual == pytest.approx(sr)
