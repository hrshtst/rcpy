import numpy as np
import pytest
import taichi as ti

from rcpy.ridge import NumpyRidge, TaichiRidge


@pytest.fixture
def setup_ridge_test():
    """A pytest fixture for Ridge solver tests."""
    X_np = np.random.rand(50, 100).astype(np.float32)
    Y_np = np.random.rand(1, 100).astype(np.float32)
    return X_np, Y_np


def test_numpy_ridge_solver(setup_ridge_test):
    """Test if the NumpyRidge solver produces a result of the correct shape."""
    X_np, Y_np = setup_ridge_test
    ridge_solver = NumpyRidge(alpha=0.1, n_iter=10)
    W_out = ridge_solver.fit(X_np, Y_np)
    assert W_out.shape == (1, 50)
    assert np.any(W_out != 0)


def test_taichi_ridge_solver(setup_ridge_test):
    """Test if the TaichiRidge solver produces a result of the correct shape."""
    ti.init(arch=ti.cpu)
    X_np, Y_np = setup_ridge_test
    ridge_solver = TaichiRidge(alpha=0.1, n_iter=10)
    W_out = ridge_solver.fit(X_np, Y_np)
    assert W_out.shape == (1, 50)
    assert np.any(W_out != 0)
