import numpy as np
import pytest
from rebasicspy.weights import initialize_weights
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix


def test_initialize_weights_get_ndarray():
    weights = initialize_weights((3, 3), sparsity_type="dense")
    assert isinstance(weights, np.ndarray)


def test_initialize_weights_get_csr():
    weights = initialize_weights((3, 3), sparsity_type="csr")
    assert isinstance(weights, csr_matrix)


def test_initialize_weights_get_csc():
    weights = initialize_weights((3, 3), sparsity_type="csc")
    assert isinstance(weights, csc_matrix)


def test_initialize_weights_get_coo():
    weights = initialize_weights((3, 3), sparsity_type="coo")
    assert isinstance(weights, coo_matrix)
