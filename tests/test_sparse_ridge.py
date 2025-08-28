import numpy as np
import pytest
import taichi as ti

# Assuming sparse_ridge.py is in the rcpy module
from rcpy.sparse_ridge import SparseTaichiRidge


@pytest.fixture
def setup_ridge_test():
    """A pytest fixture for the sparse Ridge solver test."""
    X_np = np.random.rand(50, 100).astype(np.float32)
    Y_np = np.random.rand(1, 100).astype(np.float32)
    return X_np, Y_np


def test_sparse_taichi_ridge_solver(setup_ridge_test):
    """Test if the SparseTaichiRidge solver produces a result of the correct shape."""
    ti.init(arch=ti.cpu)
    X_np, Y_np = setup_ridge_test

    # Instantiate the sparse ridge solver
    ridge_solver = SparseTaichiRidge(alpha=0.1, n_iter=10)

    # Fit the model
    W_out = ridge_solver.fit(X_np, Y_np)

    # Assertions
    assert W_out.shape == (1, 50)
    assert np.any(W_out != 0), "Output weights should not be all zeros."
