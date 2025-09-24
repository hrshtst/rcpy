# tests/test_lms.py
import numpy as np
import pytest
import taichi as ti

from rcpy.lms import NumpyLMS, TaichiLMS


@pytest.fixture
def setup_lms_test():
    """A pytest fixture for LMS solver tests."""
    np.random.seed(0)
    X_np = np.random.rand(50, 100).astype(np.float32)
    # Create a linear relationship for Y_np
    true_W_out = np.random.rand(1, 50).astype(np.float32)
    Y_np = true_W_out @ X_np + np.random.randn(1, 100) * 0.01  # Add some noise
    return X_np, Y_np, true_W_out


def test_numpy_lms_solver(setup_lms_test):
    """Test if the NumpyLMS solver produces a result of the correct shape."""
    X_np, Y_np, _ = setup_lms_test
    lms_solver = NumpyLMS(n_reservoir=50, n_output=1)
    W_out = lms_solver.fit(X_np, Y_np)
    assert W_out.shape == (1, 50)
    assert np.any(W_out != 0)


def test_taichi_lms_solver(setup_lms_test):
    """Test if the TaichiLMS solver produces a result of the correct shape."""
    ti.init(arch=ti.cpu)
    X_np, Y_np, _ = setup_lms_test
    lms_solver = TaichiLMS(n_reservoir=50, n_output=1)
    W_out = lms_solver.fit(X_np, Y_np)
    assert W_out.shape == (1, 50)
    assert np.any(W_out != 0)
