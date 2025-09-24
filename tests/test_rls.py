# tests/test_rls.py
import numpy as np
import pytest
import taichi as ti

from rcpy.rls import NumpyRLS, TaichiRLS


@pytest.fixture
def setup_rls_test():
    """A pytest fixture for RLS solver tests."""
    np.random.seed(0)
    X_np = np.random.rand(50, 100).astype(np.float32)
    # Create a linear relationship for Y_np
    true_W_out = np.random.rand(1, 50).astype(np.float32)
    Y_np = true_W_out @ X_np + np.random.randn(1, 100) * 0.01  # Add some noise
    return X_np, Y_np, true_W_out


def test_numpy_rls_solver(setup_rls_test):
    """Test if the NumpyRLS solver produces a result of the correct shape."""
    X_np, Y_np, _ = setup_rls_test
    rls_solver = NumpyRLS(n_reservoir=50, n_output=1)
    W_out = rls_solver.fit(X_np, Y_np)
    assert W_out.shape == (1, 50)
    assert np.any(W_out != 0)


def test_taichi_rls_solver(setup_rls_test):
    """Test if the TaichiRLS solver produces a result of the correct shape."""
    ti.init(arch=ti.cpu)
    X_np, Y_np, _ = setup_rls_test
    rls_solver = TaichiRLS(n_reservoir=50, n_output=1)
    W_out = rls_solver.fit(X_np, Y_np)
    assert W_out.shape == (1, 50)
    assert np.any(W_out != 0)
