import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from rebasicspy.activations import identity, relu, sigmoid, softmax, softplus, tanh


@pytest.fixture
def x():
    return np.array([0, 1, 2, -1, -2], dtype=float)


@pytest.mark.parametrize("beta", [1.0, 0.5, 1.5])
def test_softmax(x: np.ndarray, beta: float):
    y = softmax(x, beta)
    denom = np.sum(np.exp(beta * x))
    expected = np.exp(beta * x) / denom
    assert_array_almost_equal(y, expected)


@pytest.mark.parametrize("beta", [1.0, 0.5, 1.5])
def test_softplus(x: np.ndarray, beta: float):
    y = softplus(x, beta)
    e = 1.0 + np.exp(beta * x)
    expected = np.log(e) / beta
    assert_array_almost_equal(y, expected)


def test_sigmoid(x: np.ndarray):
    y = sigmoid(x)
    expected = 1.0 / (1.0 + np.exp(-x))
    assert_array_almost_equal(y, expected)


def test_tanh(x: np.ndarray):
    y = tanh(x)
    expected = np.tanh(x)
    assert_array_almost_equal(y, expected)


def test_identity(x: np.ndarray):
    y = identity(x)
    expected = x
    assert_array_equal(y, expected)


def test_relu(x: np.ndarray):
    y = relu(x)
    expected = np.maximum(0, x)
    assert_array_equal(y, expected)
