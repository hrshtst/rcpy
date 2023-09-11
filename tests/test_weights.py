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


@pytest.mark.parametrize(
    "sparsity_type,shape,connectivity", [("coo", (10, 10), 0.02), ("csr", (10, 10), 0.05), ("csc", (10, 10), 0.15)]
)
def test_initialize_weights_specify_connectivity(sparsity_type, shape: tuple[int, int], connectivity: float):
    w = initialize_weights(shape, sparsity_type=sparsity_type, connectivity=connectivity)
    actual = w.nnz / w.toarray().size
    assert actual == connectivity


@pytest.mark.parametrize("shape,connectivity", [((10, 10), 0.02), ((10, 10), 0.05), ((10, 10), 0.15)])
def test_initialize_weights_specify_connectivity_with_dense_array(shape: tuple[int, int], connectivity: float):
    w = initialize_weights(shape, sparsity_type="dense", connectivity=connectivity)
    actual = np.count_nonzero(w) / w.size
    assert actual == connectivity
