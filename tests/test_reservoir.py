# ruff: noqa: ANN401,PGH003,PLR2004,SLF001
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.sparse import csc_matrix, csr_matrix

from rcpy.activations import identity, relu, sigmoid, softmax, tanh
from rcpy.random import get_rng
from rcpy.reservoir import Reservoir, ReservoirBuilder
from rcpy.weights import normal, ones, uniform

if TYPE_CHECKING:
    from collections.abc import Callable

    from rcpy._type import SparsityType
    from rcpy.metrics import WeightsType


def get_default_builder() -> ReservoirBuilder:
    return ReservoirBuilder(reservoir_size=15, spectral_radius=0.95, connectivity=0.2, leaking_rate=0.98)


@pytest.fixture
def default_builder() -> ReservoirBuilder:
    return get_default_builder()


@pytest.fixture
def default_reservoir() -> Reservoir:
    r = Reservoir(get_default_builder())
    r.initialize_input_weights(2)
    return r


def actual_spectral_radius(w: WeightsType) -> float:
    if isinstance(w, np.ndarray):
        return max(abs(np.linalg.eigvals(w)))
    return max(abs(np.linalg.eigvals(w.toarray())))


def actual_connectivity(w: WeightsType) -> float:
    if isinstance(w, np.ndarray):
        return np.count_nonzero(w) / (w.shape[0] * w.shape[1])
    if w.shape is not None:
        return w.nnz / (w.shape[0] * w.shape[1])
    raise RuntimeError


def ensure_ndarray(w: WeightsType) -> np.ndarray:
    if isinstance(w, np.ndarray):
        return w
    return w.toarray()


class TestReservoir:
    def test_init_builder_should_be_copied(self, default_builder: ReservoirBuilder) -> None:
        res = Reservoir(default_builder)
        assert res._builder is not default_builder

    def test_init_x(self, default_reservoir: Reservoir) -> None:
        assert default_reservoir.x.shape == (15,)
        assert default_reservoir.state.shape == (15,)
        assert default_reservoir.internal_state.shape == (15,)
        assert type(default_reservoir.x) is np.ndarray
        assert type(default_reservoir.state) is np.ndarray
        assert type(default_reservoir.internal_state) is np.ndarray
        assert np.all(default_reservoir.x == 0.0)
        assert np.all(default_reservoir.state == 0.0)
        assert np.all(default_reservoir.internal_state == 0.0)

    def test_init_activation(self, default_reservoir: Reservoir) -> None:
        assert default_reservoir.activation is tanh
        assert default_reservoir.fb_activation is identity

    def test_init_noise_gain(self, default_reservoir: Reservoir) -> None:
        assert default_reservoir.noise_gain_rc == 0.0
        assert default_reservoir.noise_gain_in == 0.0
        assert default_reservoir.noise_gain_fb == 0.0
        assert default_reservoir.noise_generator.keywords["rng"] == get_rng()  # type: ignore
        assert default_reservoir.noise_generator.keywords["dist"] == "normal"  # type: ignore

    def test_init_leaking_rate(self, default_reservoir: Reservoir) -> None:
        assert default_reservoir.leaking_rate == 0.98

    def test_init_seed(self, default_reservoir: Reservoir) -> None:
        assert default_reservoir._seed is None

    def test_has_bias_input(self, default_reservoir: Reservoir) -> None:
        assert default_reservoir.has_input_bias()
        default_reservoir.initialize_input_weights(0, bias_scaling=False)
        assert not default_reservoir.has_input_bias()

    @pytest.mark.parametrize(("f", "h"), [(sigmoid, tanh), (softmax, relu)])
    def test_initialize_activation(self, f: Callable, h: Callable) -> None:
        builder = ReservoirBuilder(
            reservoir_size=5,
            spectral_radius=0.95,
            connectivity=0.2,
            leaking_rate=1.0,
            activation=f,
            fb_activation=h,
        )
        res = Reservoir(builder)
        assert res.activation is f
        assert res.fb_activation is h

    @pytest.mark.parametrize(("g_rc", "g_in", "g_fb", "dist"), [(0.1, 0.2, 0.3, "uniform"), (0.03, 0.02, 0.01, "beta")])
    def test_initialize_noise_generator(self, g_rc: float, g_in: float, g_fb: float, dist: str) -> None:
        builder = ReservoirBuilder(
            reservoir_size=5,
            spectral_radius=0.95,
            connectivity=0.2,
            leaking_rate=1.0,
            noise_gain_rc=g_rc,
            noise_gain_in=g_in,
            noise_gain_fb=g_fb,
            noise_type=dist,
        )
        res = Reservoir(builder)
        rng = get_rng()
        assert res.noise_gain_rc == g_rc
        assert res.noise_gain_in == g_in
        assert res.noise_gain_fb == g_fb
        assert res.noise_generator.keywords["rng"] is rng  # type: ignore
        assert res.noise_generator.keywords["dist"] == dist  # type: ignore

    def test_initialize_internal_weights(self, default_reservoir: Reservoir) -> None:
        W = default_reservoir.W
        assert W.shape == (15, 15)
        assert default_reservoir.size == 15

        expected_spectral_radius = default_reservoir._builder.spectral_radius
        assert actual_spectral_radius(W) == pytest.approx(expected_spectral_radius)

        expected_connectivity = default_reservoir._builder.connectivity
        assert actual_connectivity(W) == expected_connectivity

        assert type(W) is csr_matrix

    @pytest.mark.parametrize(
        ("reservoir_size", "sr", "p", "dist", "sparsity", "expected_type"),
        [(10, 1.3, 0.15, normal, "dense", np.ndarray), (20, 0.9, 0.1, uniform, "csc", csc_matrix)],
    )
    def test_initialize_internal_weights_reinitialize(
        self,
        default_reservoir: Reservoir,
        reservoir_size: int,
        sr: float,
        p: float,
        dist: Callable,
        sparsity: SparsityType,
        expected_type: Any,
    ) -> None:
        W0_array = ensure_ndarray(default_reservoir.W)
        W = default_reservoir.initialize_internal_weights(
            reservoir_size=reservoir_size,
            spectral_radius=sr,
            connectivity=p,
            W_init=dist,
            sparsity_type=sparsity,
        )
        assert W.shape == (reservoir_size, reservoir_size)
        assert default_reservoir.size == reservoir_size
        assert actual_spectral_radius(W) == pytest.approx(sr)
        assert actual_connectivity(W) == p
        assert type(W) is expected_type
        with pytest.raises(AssertionError):
            assert_array_equal(W0_array, ensure_ndarray(W))

    def test_initialize_input_weights(self, default_reservoir: Reservoir) -> None:
        input_dim = 3
        default_reservoir.initialize_input_weights(input_dim)
        Win = default_reservoir.Win
        assert Win.shape == (15, 3)
        assert default_reservoir.bias.shape == (15,)
        assert default_reservoir.has_input_bias()
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
        ("input_dim", "reservoir_size", "scaling", "p", "bias_scaling", "dist", "sparsity", "expected_type"),
        [
            (2, 10, 1.5, 0.5, 0.0, normal, "csr", csr_matrix),
            (4, 20, 0.5, 0.2, 0.1, uniform, "csc", csc_matrix),
        ],
    )
    def test_initialize_input_weights_reinitialize(
        self,
        default_reservoir: Reservoir,
        input_dim: int,
        reservoir_size: int,
        scaling: float,
        p: float,
        bias_scaling: float,
        dist: Callable,
        sparsity: SparsityType,
        expected_type: Any,
    ) -> None:
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
        assert default_reservoir.has_input_bias() is (bias_scaling != 0.0)
        if bias_scaling:
            assert np.any(default_reservoir.bias != 0.0)
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

        # Check if Win is an expected sparsity type
        assert type(Win) is expected_type

    def test_initialize_input_weights_check_bias_scaling(self, default_reservoir: Reservoir) -> None:
        input_dim = 3
        Win = default_reservoir.initialize_input_weights(input_dim, bias_scaling=0.5)
        assert Win.shape == (default_reservoir.size, input_dim)

        # the first column of weights must between [-0.5, 0.5]
        assert default_reservoir.has_input_bias()
        assert np.all(default_reservoir.bias < 0.5)
        assert np.all(default_reservoir.bias > -0.5)
        assert type(default_reservoir.bias) is np.ndarray

        # the rest of weights may be > 0.5 or < -0.5
        assert np.any(Win > 0.5)
        assert np.any(Win < -0.5)

    @pytest.mark.parametrize("bias", [0.0, False])
    def test_initialize_input_weights_when_bias_is_zero(self, default_reservoir: Reservoir, bias: float | bool) -> None:
        input_dim = 2
        Win = default_reservoir.initialize_input_weights(input_dim, bias_scaling=bias)
        assert Win.shape == (default_reservoir.size, input_dim)

        # When bias is zero or False, bias vector should be zero.
        initialized_bias = default_reservoir.bias
        assert not default_reservoir.has_input_bias()
        assert initialized_bias.shape == (default_reservoir.size,)
        assert np.all(initialized_bias == 0.0)
        assert type(initialized_bias) is np.ndarray

    def test_initialize_input_weights_allow_zero_dimension(self, default_reservoir: Reservoir) -> None:
        Nx = default_reservoir._builder.reservoir_size
        Win = default_reservoir.initialize_input_weights(0)
        assert Win.shape == (Nx, 0)
        assert type(Win) is np.ndarray

        u = np.empty((0, 1))
        r = Win @ u
        assert r.shape == (Nx, 1)
        assert np.all(r == 0.0)

        bias = default_reservoir.bias
        assert bias.shape == (default_reservoir._builder.reservoir_size,)
        assert np.all(bias != 0.0)
        assert type(bias) is np.ndarray

    @pytest.mark.parametrize(
        ("input_dim", "scaling"),
        [
            (2, (0.1, 0.2)),
            (2, (0.5, 1.0)),
            (3, (0.1, 0.2, 0.3)),
            (3, (0.5, 1.0, 1.5)),
            (4, (0.1, 0.2, 0.3, 0.4)),
            (4, (0.5, 1.0, 1.5, 2.0)),
        ],
    )
    def test_initialize_input_weights_allow_element_wise_scaling(
        self,
        default_reservoir: Reservoir,
        input_dim: int,
        scaling: tuple[int, ...],
    ) -> None:
        size = default_reservoir.size
        Win = default_reservoir.initialize_input_weights(input_dim, input_scaling=scaling, Win_init=ones)
        assert Win.shape == (size, input_dim)
        Win_array = ensure_ndarray(Win)
        E = np.ones(size, dtype=float)
        for i, s in enumerate(scaling):
            Win_i = Win_array[:, i]
            assert_array_equal(Win_i, s * E)

    @pytest.mark.parametrize(
        ("input_dim", "scaling"),
        [
            (3, (0.1, 0.2)),
            (2, (0.1, 0.2, 0.3)),
            (1, (0.1, 0.2, 0.3, 0.4)),
        ],
    )
    def test_initialize_input_weights_raise_exception_when_input_scaling_size_mismatch(
        self,
        default_reservoir: Reservoir,
        input_dim: int,
        scaling: tuple[int, ...],
    ) -> None:
        pattern = "The size of `input_scaling` is mismatched with `input_dim`."
        with pytest.raises(ValueError, match=pattern) as exinfo:
            _ = default_reservoir.initialize_input_weights(input_dim, input_scaling=scaling)
        assert str(exinfo.value) == "The size of `input_scaling` is mismatched with `input_dim`."

    def test_raise_exception_when_access_Win_before_initialization(self) -> None:
        res = Reservoir(get_default_builder())
        with pytest.raises(RuntimeError):
            _ = res.Win

    def test_raise_exception_when_access_bias_before_initialization(self) -> None:
        res = Reservoir(get_default_builder())
        with pytest.raises(RuntimeError):
            _ = res.bias

    def test_kernel(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = default_reservoir.input_dim
        W = default_reservoir.W
        Win = default_reservoir.Win
        bias = default_reservoir.bias
        u = np.ones(input_dim)
        x = np.zeros(size)

        expected = Win @ u + W @ x + bias
        actual = default_reservoir.kernel(u, x, None)

        assert_array_almost_equal(expected, actual)

    def test_kernel_zero_input(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = 0
        W = default_reservoir.W
        Win = default_reservoir.initialize_input_weights(input_dim)
        bias = default_reservoir.bias
        u = np.ones(input_dim)
        x = np.zeros(size)

        expected = Win @ u + W @ x + bias
        actual = default_reservoir.kernel(u, x, None)

        assert_array_almost_equal(expected, actual)
        assert_array_almost_equal(bias, actual)

    def test_kernel_input_noise(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = default_reservoir.input_dim
        W = default_reservoir.W
        Win = default_reservoir.Win
        bias = default_reservoir.bias
        u = np.ones(input_dim)
        x = np.zeros(size)

        default_reservoir.initialize_noise_generator(noise_gain_in=0.01)
        no_noise = Win @ u + W @ x + bias
        actual = default_reservoir.kernel(u, x, None)

        assert not np.any(no_noise == actual)

    def test_forward_internal(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = default_reservoir.input_dim
        u = np.ones(input_dim)
        x = np.zeros(size)
        lr = default_reservoir.leaking_rate

        pre_x = default_reservoir.kernel(u, x, None)
        expected = (1 - lr) * x + lr * np.tanh(pre_x)
        actual = default_reservoir.forward_internal(u, None)

        assert_array_almost_equal(expected, actual)
        assert_array_almost_equal(expected, default_reservoir.state)

    def test_forward_internal_zero_input(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = 0
        default_reservoir.initialize_input_weights(input_dim)
        u = np.ones(input_dim)
        x = np.zeros(size)
        lr = default_reservoir.leaking_rate

        pre_x = default_reservoir.kernel(u, x, None)
        expected = (1 - lr) * x + lr * np.tanh(pre_x)
        actual = default_reservoir.forward_internal(u, None)

        assert_array_almost_equal(expected, actual)
        assert_array_almost_equal(expected, default_reservoir.state)

    def test_forward_internal_rc_noise(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = default_reservoir.input_dim
        u = np.ones(input_dim)
        x = np.zeros(size)
        lr = default_reservoir.leaking_rate

        default_reservoir.initialize_noise_generator(noise_gain_rc=0.01)
        pre_x = default_reservoir.kernel(u, x, None)
        no_noise = (1 - lr) * x + lr * np.tanh(pre_x)
        actual = default_reservoir.forward_internal(u, None)

        assert not np.any(no_noise == actual)

    def test_forward_external(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = default_reservoir.input_dim
        u = np.ones(input_dim)
        x = np.zeros(size)
        s = np.zeros(size)
        lr = default_reservoir.leaking_rate

        pre_x = default_reservoir.kernel(u, x, None)
        s_next = (1 - lr) * s + lr * pre_x
        x_next = np.tanh(s_next)
        actual = default_reservoir.forward_external(u, None)

        assert_array_almost_equal(x_next, actual)
        assert_array_almost_equal(x_next, default_reservoir.state)
        assert_array_equal(s_next, default_reservoir.internal_state)

    def test_forward_external_zero_input(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = 0
        default_reservoir.initialize_input_weights(input_dim)
        u = np.ones(input_dim)
        x = np.zeros(size)
        s = np.zeros(size)
        lr = default_reservoir.leaking_rate

        pre_x = default_reservoir.kernel(u, x, None)
        s_next = (1 - lr) * s + lr * pre_x
        x_next = np.tanh(s_next)
        actual = default_reservoir.forward_external(u, None)

        assert_array_almost_equal(x_next, actual)
        assert_array_almost_equal(x_next, default_reservoir.state)
        assert_array_equal(s_next, default_reservoir.internal_state)

    def test_forward_external_rc_noise(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = default_reservoir.input_dim
        u = np.ones(input_dim)
        x = np.zeros(size)
        s = np.zeros(size)
        lr = default_reservoir.leaking_rate

        default_reservoir.initialize_noise_generator(noise_gain_rc=0.01)
        pre_x = default_reservoir.kernel(u, x, None)
        s_next = (1 - lr) * s + lr * pre_x
        x_next = np.tanh(s_next)
        actual = default_reservoir.forward_external(u, None)

        assert not np.any(s_next == default_reservoir.internal_state)
        assert not np.any(x_next == actual)

    def test_has_feedback(self, default_reservoir: Reservoir) -> None:
        assert not default_reservoir.has_feedback()

    def test_raise_exception_when_access_Wfb_before_init(self, default_reservoir: Reservoir) -> None:
        with pytest.raises(RuntimeError):
            _ = default_reservoir.Wfb

    def test_initialize_feedback_weights(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        output_dim = 3
        default_reservoir.initialize_feedback_weights(output_dim)
        Wfb = default_reservoir.Wfb
        assert Wfb.shape == (size, output_dim)
        assert default_reservoir.has_feedback()

        # Check if Wfb is sampled between [-1, 1]
        assert np.all(Wfb < 1)
        assert np.all(Wfb > -1)
        assert np.any(Wfb < 0)
        assert np.any(Wfb > 0)

        # Check if Wfb is a dense matrix
        assert not np.any(Wfb == 0.0)
        assert type(Wfb) is np.ndarray

    @pytest.mark.parametrize(
        ("output_dim", "reservoir_size", "scaling", "p", "dist", "sparsity", "expected_type"),
        [
            (2, 10, 1.5, 0.5, normal, "csr", csr_matrix),
            (4, 20, 0.5, 0.2, uniform, "csc", csc_matrix),
        ],
    )
    def test_initialize_feedback_weights_reinitialize(
        self,
        default_reservoir: Reservoir,
        output_dim: int,
        reservoir_size: int,
        scaling: float,
        p: float,
        dist: Callable,
        sparsity: SparsityType,
        expected_type: Any,
    ) -> None:
        default_reservoir.initialize_feedback_weights(
            output_dim,
            reservoir_size=reservoir_size,
            fb_scaling=scaling,
            fb_connectivity=p,
            Wfb_init=dist,
            sparsity_type=sparsity,
        )
        Wfb = default_reservoir.Wfb
        assert Wfb.shape == (reservoir_size, output_dim)
        assert default_reservoir.has_feedback()

        # Check if Wfb is scaled by fb_scaling
        Wfb_array = Wfb.toarray()  # type: ignore
        if dist == uniform:
            assert np.all(Wfb_array < scaling)
            assert np.all(Wfb_array > -scaling)
        assert np.any(Wfb_array < 0)
        assert np.any(Wfb_array > 0)

        # Check if Wfb is an expected sparsity type
        assert type(Wfb) is expected_type

    @pytest.mark.parametrize(
        ("output_dim", "scaling"),
        [
            (2, (0.1, 0.2)),
            (2, (0.5, 1.0)),
            (3, (0.1, 0.2, 0.3)),
            (3, (0.5, 1.0, 1.5)),
            (4, (0.1, 0.2, 0.3, 0.4)),
            (4, (0.5, 1.0, 1.5, 2.0)),
        ],
    )
    def test_initialize_feedback_weights_allow_element_wise_scaling(
        self,
        default_reservoir: Reservoir,
        output_dim: int,
        scaling: tuple[int, ...],
    ) -> None:
        size = default_reservoir.size
        Wfb = default_reservoir.initialize_feedback_weights(output_dim, fb_scaling=scaling, Wfb_init=ones)
        assert Wfb.shape == (size, output_dim)
        assert default_reservoir.has_feedback()
        Wfb_array = ensure_ndarray(Wfb)
        E = np.ones(size, dtype=float)
        for i, s in enumerate(scaling):
            Wfb_i = Wfb_array[:, i]
            assert_array_equal(Wfb_i, s * E)

    @pytest.mark.parametrize(
        ("output_dim", "scaling"),
        [
            (3, (0.1, 0.2)),
            (2, (0.1, 0.2, 0.3)),
            (1, (0.1, 0.2, 0.3, 0.4)),
        ],
    )
    def test_initialize_feedback_weights_raise_exception_when_fb_scaling_size_mismatch(
        self,
        default_reservoir: Reservoir,
        output_dim: int,
        scaling: tuple[int, ...],
    ) -> None:
        pattern = "The size of `fb_scaling` is mismatched with `output_dim`."
        with pytest.raises(ValueError, match=pattern) as exinfo:
            _ = default_reservoir.initialize_feedback_weights(output_dim, fb_scaling=scaling)
        assert str(exinfo.value) == "The size of `fb_scaling` is mismatched with `output_dim`."

    def test_kernel_with_feedback(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = default_reservoir.input_dim
        output_dim = 3
        default_reservoir.initialize_feedback_weights(output_dim)

        W = default_reservoir.W
        Win = default_reservoir.Win
        Wfb = default_reservoir.Wfb
        bias = default_reservoir.bias
        u = np.ones(input_dim)
        x = np.zeros(size)
        y = np.ones(output_dim)

        expected = Win @ u + W @ x + bias + Wfb @ y
        actual = default_reservoir.kernel(u, x, y)

        assert_array_almost_equal(expected, actual)

    def test_kernel_with_feedback_noise(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = default_reservoir.input_dim
        output_dim = 3
        default_reservoir.initialize_feedback_weights(output_dim)

        W = default_reservoir.W
        Win = default_reservoir.Win
        Wfb = default_reservoir.Wfb
        bias = default_reservoir.bias
        u = np.ones(input_dim)
        x = np.zeros(size)
        y = np.ones(output_dim)

        default_reservoir.initialize_noise_generator(noise_gain_fb=0.01)
        no_noise = Win @ u + W @ x + bias + Wfb @ y
        actual = default_reservoir.kernel(u, x, y)

        assert not np.any(no_noise == actual)

    def test_forward_internal_with_feedback(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = default_reservoir.input_dim
        output_dim = 3
        default_reservoir.initialize_feedback_weights(output_dim)

        u = np.ones(input_dim)
        x = np.zeros(size)
        y = np.ones(output_dim)
        lr = default_reservoir.leaking_rate

        pre_x = default_reservoir.kernel(u, x, y)
        expected = (1 - lr) * x + lr * np.tanh(pre_x)
        actual = default_reservoir.forward_internal(u, y)

        assert_array_almost_equal(expected, actual)
        assert_array_almost_equal(expected, default_reservoir.state)

    def test_forward_external_with_feedback(self, default_reservoir: Reservoir) -> None:
        size = default_reservoir.size
        input_dim = default_reservoir.input_dim
        output_dim = 3
        default_reservoir.initialize_feedback_weights(output_dim)

        u = np.ones(input_dim)
        x = np.zeros(size)
        y = np.ones(output_dim)
        s = np.zeros(size)
        lr = default_reservoir.leaking_rate

        pre_x = default_reservoir.kernel(u, x, y)
        s_next = (1 - lr) * s + lr * pre_x
        x_next = np.tanh(s_next)
        actual = default_reservoir.forward_external(u, y)

        assert_array_almost_equal(x_next, actual)
        assert_array_almost_equal(x_next, default_reservoir.state)
        assert_array_equal(s_next, default_reservoir.internal_state)

    def test_forward_raise_exception_when_y_is_given_before_init(self, default_reservoir: Reservoir) -> None:
        input_dim = default_reservoir.input_dim
        output_dim = 3
        u = np.ones(input_dim)
        y = np.ones(output_dim)

        assert not default_reservoir.has_feedback()
        # Raise exception when y is given before initializing Wfb.
        with pytest.raises(RuntimeError) as exinfo:
            _ = default_reservoir.forward(u, y)
        assert str(exinfo.value).startswith("Feedback weights have not been initialized yet.")

    def test_forward_raise_exception_when_y_is_not_given_after_init(self, default_reservoir: Reservoir) -> None:
        input_dim = default_reservoir.input_dim
        output_dim = 3
        u = np.ones(input_dim)

        # Raise exception when y is not given after initializing Wfb.
        default_reservoir.initialize_feedback_weights(output_dim)
        with pytest.raises(RuntimeError) as exinfo:
            _ = default_reservoir.forward(u)
        assert str(exinfo.value).startswith("Reservoir has feedback connection, but no feedback signal was given.")


# Local Variables:
# jinx-local-words: "Wfb csc csr fb noqa rc rng sr"
# End:
