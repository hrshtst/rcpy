import numpy as np
import pytest
import taichi as ti

# Assuming sparse_esn.py is in the rcpy module
from rcpy.config import get_config
from rcpy.sparse_esn import SparseEchoStateNetwork


@pytest.fixture
def setup_sparse_esn_test():
    """A pytest fixture for Sparse ESN tests."""
    conf = get_config()
    conf.esn.n_reservoir = 50
    # Ensure n_train_samples > washout_period to avoid zero-sized arrays
    conf.data.n_train_samples = 200
    conf.data.washout_period = 100
    conf.esn.sparsity = 0.9  # Use a high sparsity for testing

    train_input = np.random.rand(conf.data.n_train_samples, conf.esn.n_input).astype(np.float32)
    train_target = np.random.rand(conf.data.n_train_samples, conf.esn.n_output).astype(np.float32)
    test_input = np.random.rand(50, conf.esn.n_input).astype(np.float32)

    return conf, train_input, train_target, test_input


def test_sparse_esn_initialization(setup_sparse_esn_test):
    """Test if the Sparse ESN initializes its weights and sparse matrix correctly."""
    ti.init(arch=ti.cpu)
    conf, _, _, _ = setup_sparse_esn_test
    esn = SparseEchoStateNetwork(conf)

    # Check that the COO fields have been created
    assert esn.W_res_rows is not None
    assert esn.W_res_cols is not None
    assert esn.W_res_vals is not None
    assert isinstance(esn.W_res_vals, ti.Field)

    # Check that the number of non-zero elements is reasonable
    expected_nnz = int(50 * 50 * (1 - 0.9))
    assert esn.W_res_vals.shape[0] >= expected_nnz * 0.8
    assert esn.W_res_vals.shape[0] <= expected_nnz * 1.2


def test_sparse_esn_fit_predict(setup_sparse_esn_test):
    """Test if the Sparse ESN can fit and predict without errors."""
    ti.init(arch=ti.cpu)
    conf, train_input, train_target, test_input = setup_sparse_esn_test
    esn = SparseEchoStateNetwork(conf)

    w_out_before = esn.W_out.to_numpy()
    assert np.all(w_out_before == 0)

    esn.fit(train_input, train_target, conf)

    w_out_after = esn.W_out.to_numpy()
    assert np.any(w_out_after != 0), "Output weights should be updated after fitting."

    predictions = esn.predict(test_input, conf)
    assert predictions.shape == (50, conf.esn.n_output)
