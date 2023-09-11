import numpy as np
import pytest
from rebasicspy.weights import initialize_weights
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix


@pytest.mark.parametrize(
    "sparsity_type,expected_type",
    [("dense", np.ndarray), ("csr", csr_matrix), ("csc", csc_matrix), ("coo", coo_matrix)],
)
def test_initialize_weights_specify_sparsity_type(sparsity_type, expected_type):
    w = initialize_weights((5, 5), sparsity_type=sparsity_type)
    assert isinstance(w, expected_type)
