import warnings
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from rebasicspy.random import get_rng
from rebasicspy.weights import WeightsType, bernoulli, initialize_weights, normal, ones, sparse_random, uniform, zeros
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix


def actual_spectral_radius(w: WeightsType) -> float:
    if isinstance(w, np.ndarray):
        return max(abs(np.linalg.eigvals(w)))
    else:
        return max(abs(np.linalg.eigvals(w.toarray())))


def actual_connectivity(w: WeightsType) -> float:
    if isinstance(w, np.ndarray):
        return np.count_nonzero(w) / (w.shape[0] * w.shape[1])
    else:
        return w.nnz / (w.shape[0] * w.shape[1])


def ensure_ndarray(w: WeightsType) -> np.ndarray:
    if isinstance(w, np.ndarray):
        return w
    else:
        return w.toarray()


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
def test_sparse_random_if_connectivity_is_none_then_it_should_be_dense_array(sparsity_type):
    default = 1.0
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
    w = normal((20, 20), loc=loc, scale=scale, connectivity=1.0, seed=get_rng(123))
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
    assert actual == pytest.approx(sr, rel=1e-2)


@pytest.mark.parametrize("sr", [0.95, 1.0, 1.3])
def test_initialize_weights_scale_spectral_radius_with_dense_array(sr: float):
    w = initialize_weights((100, 100), uniform, spectral_radius=sr, sparsity_type="dense")
    actual = max(abs(np.linalg.eigvals(w)))
    assert isinstance(w, np.ndarray)
    assert actual == pytest.approx(sr, rel=1e-2)


@pytest.mark.parametrize("scaling", [0.2, 5.0])
def test_initialize_weights_scale_inputs(scaling: float):
    w = initialize_weights((10, 1), uniform, scaling=scaling, sparsity_type="dense", connectivity=1.0)
    if scaling > 1.0:
        assert np.max(w) > 1.0
        assert np.min(w) < -1.0
    else:
        assert np.max(w) < scaling
        assert np.min(w) > -scaling


@pytest.mark.parametrize(
    "size,in_dim,scaling,sparsity",
    [
        (5, 2, 0.5, "dense"),
        (6, 1, 0.1, "csc"),
        (6, 3, (0.1, 0.2, 0.3), "csr"),
        (4, 1, (0.4,), "coo"),
    ],
)
def test_initialize_weights_scaling(size, in_dim, scaling, sparsity):
    w = initialize_weights((size, in_dim), ones, scaling=scaling, sparsity_type=sparsity)
    w_arr = ensure_ndarray(w)

    if isinstance(scaling, float):
        expected_w = np.ones((size, in_dim)) * np.array([scaling])
    else:
        expected_w = np.ones((size, in_dim)) * np.array(scaling)
    assert_array_equal(w_arr, expected_w)


@pytest.mark.parametrize(
    "size,in_dim,scaling,sparsity,expected_type",
    [
        (5, 2, 0.5, "dense", np.ndarray),
        (6, 1, 0.1, "csc", csc_matrix),
        (6, 3, (0.1, 0.2, 0.3), "csr", csr_matrix),
        (4, 1, (0.4,), "coo", csr_matrix),  # coo_matrix -> csr_matrix
    ],
)
def test_initialize_weights_scaling_check_type(size, in_dim, scaling, sparsity, expected_type):
    w = initialize_weights((size, in_dim), uniform, scaling=scaling, sparsity_type=sparsity)
    w_arr = ensure_ndarray(w)
    assert np.all(w_arr != 0.0)
    assert type(w) is expected_type


@pytest.mark.parametrize(
    "size,in_dim,scaling,sparsity",
    [
        (5, 2, (0.1, 0.2, 0.3), "dense"),
        (5, 1, (0.1, 0.2), "csc"),
        (6, 3, (0.1, 0.2), "csr"),
        (4, 2, (0.4,), "coo"),
    ],
)
def test_initialize_weights_raise_exception_when_shape_of_scaling_incorrect(size, in_dim, scaling, sparsity):
    with pytest.raises(ValueError):
        _ = initialize_weights((size, in_dim), uniform, scaling=scaling, sparsity_type=sparsity)
