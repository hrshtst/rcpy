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
        assert default_reservoir.size == 15

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
        assert default_reservoir.size == reservoir_size
        assert actual_spectral_radius(W) == pytest.approx(sr)
        assert actual_connectivity(W) == p
        assert type(W) is expected_type
        with pytest.raises(AssertionError):
            assert_array_equal(W0_array, ensure_ndarray(W))

    def test_initialize_input_weights(self, default_reservoir: Reservoir):
        input_dim = 3
        default_reservoir.initialize_input_weights(input_dim)
        Win = default_reservoir.Win
        assert Win.shape == (15, 3)
        assert default_reservoir.bias.shape == (15,)
        assert default_reservoir.has_input_bias
        assert np.all(default_reservoir.bias != 0.0)
        assert type(default_reservoir.bias) is np.ndarray

        # Check if Win is sampled between [-1, 1]
        assert np.all(Win < 1)
        assert np.all(Win > -1)
        assert np.any(Win < 0)
        assert np.any(Win > 0)

        # Check if Win is a dense matrix
        assert not np.any(Win == 0.0)
        assert type(Win) is np.ndarray

    @pytest.mark.parametrize(
        "input_dim,reservoir_size,scaling,p,bias_scaling,dist,sparsity,expected_type",
        [
            (2, 10, 1.5, 0.5, 0.0, normal, "csr", csr_matrix),
            (4, 20, 0.5, 0.2, 0.1, uniform, "csc", csc_matrix),
        ],
    )
    def test_initialize_input_weights_reinitialize(
        self,
        default_reservoir: Reservoir,
        input_dim,
        reservoir_size,
        scaling,
        p,
        bias_scaling,
        dist,
        sparsity,
        expected_type,
    ):
        default_reservoir.initialize_input_weights(
            input_dim,
            reservoir_size=reservoir_size,
            input_scaling=scaling,
            input_connectivity=p,
            bias_scaling=bias_scaling,
            Win_init=dist,
            sparsity_type=sparsity,
        )
        Win = default_reservoir.Win
        assert Win.shape == (reservoir_size, input_dim)

        assert default_reservoir.bias.shape == (reservoir_size,)
        assert default_reservoir.has_input_bias is (bias_scaling != 0.0)
        if bias_scaling:
            assert np.all(default_reservoir.bias != 0.0)
        else:
            assert_array_equal(default_reservoir.bias, np.zeros(reservoir_size))
        assert type(default_reservoir.bias) is np.ndarray

        # Check if Win is scaled by input_scaling
        Win_array = Win.toarray()  # type: ignore
        if dist == uniform:
            assert np.all(Win_array < scaling)
            assert np.all(Win_array > -scaling)
        assert np.any(Win_array < 0)
        assert np.any(Win_array > 0)

        # Check if Win is an expected sparity type
        assert type(Win) is expected_type

    def test_initialize_input_weights_check_bias_scaling(self, default_reservoir: Reservoir):
        input_dim = 3
        Win = default_reservoir.initialize_input_weights(input_dim, bias_scaling=0.5)
        assert Win.shape == (default_reservoir.size, input_dim)

        # the first column of weights must between [-0.5, 0.5]
        assert default_reservoir.has_input_bias
        assert np.all(default_reservoir.bias < 0.5)
        assert np.all(default_reservoir.bias > -0.5)
        assert type(default_reservoir.bias) is np.ndarray

        # the rest of weights may be > 0.5 or < -0.5
        assert np.any(Win > 0.5)
        assert np.any(Win < -0.5)
