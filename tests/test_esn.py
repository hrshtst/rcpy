import warnings

import numpy as np
import pytest
import taichi as ti

from rcpy.config import get_config
from rcpy.esn import EchoStateNetwork, NumpyEchoStateNetwork

# Ignore DeprecationWarning from external libraries like Taichi
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture
def setup_esn_test():
    """A pytest fixture for ESN tests."""
    conf = get_config()
    conf.esn.n_reservoir = 50
    # Ensure n_train_samples > washout_period to avoid zero-sized arrays
    conf.data.n_train_samples = 200
    conf.data.washout_period = 100

    train_input = np.random.rand(conf.data.n_train_samples, conf.esn.n_input).astype(np.float32)
    train_target = np.random.rand(conf.data.n_train_samples, conf.esn.n_output).astype(np.float32)
    test_input = np.random.rand(50, conf.esn.n_input).astype(np.float32)

    return conf, train_input, train_target, test_input


def test_numpy_esn_initialization(setup_esn_test):
    """Test if the NumPy ESN initializes its weights correctly."""
    conf, _, _, _ = setup_esn_test
    esn = NumpyEchoStateNetwork(conf)

    assert esn.W_in is not None
    assert esn.W_res is not None
    assert esn.W_res.shape == (50, 50)
    assert esn.x.shape == (50,)


def test_numpy_esn_fit_predict(setup_esn_test):
    """Test if the NumPy ESN can fit and predict without errors."""
    conf, train_input, train_target, test_input = setup_esn_test
    esn = NumpyEchoStateNetwork(conf)

    assert esn.W_out is None
    esn.fit(train_input, train_target, conf)
    assert esn.W_out is not None
    assert esn.W_out.shape == (conf.esn.n_output, conf.esn.n_reservoir)

    predictions = esn.predict(test_input, conf)
    assert predictions.shape == (50, conf.esn.n_output)


def test_taichi_esn_initialization(setup_esn_test):
    """Test if the Taichi ESN initializes its fields correctly."""
    ti.init(arch=ti.cpu)
    conf, _, _, _ = setup_esn_test
    esn = EchoStateNetwork(conf)

    assert esn.W_res.shape == (50, 50)
    assert esn.x.shape == (50,)

    w_res_np = esn.W_res.to_numpy()
    assert np.any(w_res_np != 0)


def test_taichi_esn_fit_predict(setup_esn_test):
    """Test if the Taichi ESN can fit and predict without errors."""
    ti.init(arch=ti.cpu)
    conf, train_input, train_target, test_input = setup_esn_test
    esn = EchoStateNetwork(conf)

    w_out_before = esn.W_out.to_numpy()
    assert np.all(w_out_before == 0)

    esn.fit(train_input, train_target, conf)

    w_out_after = esn.W_out.to_numpy()
    assert np.any(w_out_after != 0)

    predictions = esn.predict(test_input, conf)
    assert predictions.shape == (50, conf.esn.n_output)
