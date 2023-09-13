import numpy as np
import pytest
from numpy.testing import assert_array_equal
from rebasicspy.metrics import WeightsType
from rebasicspy.reservoir import Reservoir, ReservoirBuilder
from rebasicspy.weights import normal, uniform
from scipy.sparse import csc_matrix, csr_matrix


def get_default_builder() -> ReservoirBuilder:
    return ReservoirBuilder(reservoir_size=15, spectral_radius=0.95, connectivity=0.2, leaking_rate=0.98)


@pytest.fixture
def default_builder() -> ReservoirBuilder:
    return get_default_builder()


@pytest.fixture
def default_reservoir() -> Reservoir:
    return Reservoir(get_default_builder())


def actual_spectral_radius(w: WeightsType) -> float:
    if isinstance(w, np.ndarray):
        return max(abs(np.linalg.eigvals(w)))
    else:
        return max(abs(np.linalg.eigvals(w.toarray())))


def actual_connectivity(w: WeightsType) -> float:
    if isinstance(w, np.ndarray):
        return np.count_nonzero(w) / (w.shape[0] * w.shape[1])
    else:
        return w.nnz / (w.shape[0] * w.shape[1])


def ensure_ndarray(w: WeightsType) -> np.ndarray:
    if isinstance(w, np.ndarray):
        return w
    else:
        return w.toarray()


class TestReservoir:
    def test_init_builder_should_be_copied(self, default_builder: ReservoirBuilder):
        res = Reservoir(default_builder)
        assert res._builder is not default_builder

    def test_init_leaking_rate(self, default_reservoir: Reservoir):
        assert default_reservoir.leaking_rate == 0.98

    def test_init_seed(self, default_reservoir: Reservoir):
        assert default_reservoir._seed is None

    def test_initialize_internal_weights(self, default_reservoir: Reservoir):
        W = default_reservoir.W
        assert W.shape == (15, 15)

        expected_spectral_radius = default_reservoir._builder.spectral_radius
        assert actual_spectral_radius(W) == pytest.approx(expected_spectral_radius)

        expected_connectivity = default_reservoir._builder.connectivity
        assert actual_connectivity(W) == expected_connectivity

        assert type(W) is csr_matrix

    @pytest.mark.parametrize(
        "reservoir_size,sr,p,dist,sparsity,expected_type",
        [(10, 1.3, 0.15, normal, "dense", np.ndarray), (20, 0.9, 0.1, uniform, "csc", csc_matrix)],
    )
    def test_initialize_internal_weights_reinitialize(
        self, default_reservoir: Reservoir, reservoir_size, sr, p, dist, sparsity, expected_type
    ):
        W0_array = ensure_ndarray(default_reservoir.W)
        W = default_reservoir.initialize_internal_weights(
            reservoir_size=reservoir_size, spectral_radius=sr, connectivity=p, W_init=dist, sparsity_type=sparsity
        )
        assert W.shape == (reservoir_size, reservoir_size)
        assert actual_spectral_radius(W) == pytest.approx(sr)
        assert actual_connectivity(W) == p
        assert type(W) is expected_type
        with pytest.raises(AssertionError):
            assert_array_equal(W0_array, ensure_ndarray(W))
