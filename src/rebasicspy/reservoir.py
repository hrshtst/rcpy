import copy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable

import numpy as np

from rebasicspy._type import SparsityType, WeightsType
from rebasicspy.activations import identity, tanh
from rebasicspy.random import get_rng, noise
from rebasicspy.weights import initialize_weights, uniform


@dataclass
class ReservoirBuilder:
    reservoir_size: int
    spectral_radius: float
    connectivity: float
    leaking_rate: float
    W_init: Callable[..., WeightsType] = uniform
    Win_init: Callable[..., WeightsType] = uniform
    input_scaling: float | Iterable[float] = 1.0
    input_connectivity: float = 1.0
    bias_scaling: float = 1.0
    activation: Callable[[np.ndarray], np.ndarray] = tanh
    fb_activation: Callable[[np.ndarray], np.ndarray] = identity
    noise_gain_rc: float = 0.0
    noise_gain_in: float = 0.0
    noise_gain_fb: float = 0.0
    noise_type: str = "normal"
    seed: int | None = None


_ERR_MSG_INPUT_WEIGHTS_NOT_INITIALIZED = (
    f"Input weights have not been initialized yet. Call `initialize_input_weights` first."
)


class Reservoir(object):
    _builder: ReservoirBuilder
    _x: np.ndarray
    _W: WeightsType
    _Win: WeightsType
    _bias: np.ndarray
    _leaking_rate: float
    _activation: Callable[[np.ndarray], np.ndarray]
    _fb_activation: Callable[[np.ndarray], np.ndarray]
    _noise_gain_rc: float
    _noise_gain_in: float
    _noise_gain_fb: float
    _noise_generator: Callable[..., np.ndarray]
    _seed: int | None

    def __init__(self, builder: ReservoirBuilder):
        self._builder = copy.copy(builder)
        self._leaking_rate = self._builder.leaking_rate
        self._seed = self._builder.seed
        self.initialize_activation()
        self.initialize_noise_generator()
        self.initialize_reservoir_state()
        self.initialize_internal_weights()

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def W(self) -> WeightsType:
        return self._W

    @property
    def Win(self) -> WeightsType:
        if not hasattr(self, "_Win"):
            raise RuntimeError(_ERR_MSG_INPUT_WEIGHTS_NOT_INITIALIZED)
        return self._Win

    @property
    def bias(self) -> np.ndarray:
        if not hasattr(self, "_bias"):
            raise RuntimeError(_ERR_MSG_INPUT_WEIGHTS_NOT_INITIALIZED)
        return self._bias

    @property
    def has_input_bias(self) -> bool:
        if hasattr(self, "_bias"):
            return not np.all(self._bias == 0.0)
        else:
            return self._builder.bias_scaling != 0.0

    @property
    def leaking_rate(self) -> float:
        return self._leaking_rate

    @property
    def size(self) -> int:
        return self._W.shape[0]

    @property
    def input_dim(self) -> int:
        return self.Win.shape[1]

    @property
    def activation(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._activation

    @property
    def fb_activation(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._fb_activation

    @property
    def noise_gain_rc(self) -> float:
        return self._noise_gain_rc

    @property
    def noise_gain_in(self) -> float:
        return self._noise_gain_in

    @property
    def noise_gain_fb(self) -> float:
        return self._noise_gain_fb

    @property
    def noise_generator(self) -> Callable[..., np.ndarray]:
        return self._noise_generator

    def initialize_activation(
        self,
        activation: Callable[[np.ndarray], np.ndarray] | None = None,
        fb_activation: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        if activation is None:
            activation = self._builder.activation
        if fb_activation is None:
            fb_activation = self._builder.fb_activation

        self._activation = activation
        self._fb_activation = fb_activation

    def initialize_noise_generator(
        self,
        noise_gain_rc: float | None = None,
        noise_gain_in: float | None = None,
        noise_gain_fb: float | None = None,
        noise_type: str | None = None,
        seed: int | None = None,
    ):
        if noise_gain_rc is None:
            noise_gain_rc = self._builder.noise_gain_rc
        if noise_gain_in is None:
            noise_gain_in = self._builder.noise_gain_in
        if noise_gain_fb is None:
            noise_gain_fb = self._builder.noise_gain_fb
        if noise_type is None:
            noise_type = self._builder.noise_type
        if seed is None:
            seed = self._builder.seed

        rng = get_rng(seed)
        self._noise_gain_rc = noise_gain_rc
        self._noise_gain_in = noise_gain_in
        self._noise_gain_fb = noise_gain_fb
        self._noise_generator = partial(noise, rng=rng, dist=noise_type)

    def initialize_reservoir_state(self, reservoir_size: int | None = None) -> np.ndarray:
        if reservoir_size is None:
            reservoir_size = self._builder.reservoir_size

        self._x = np.zeros(reservoir_size)
        return self._x

    def initialize_internal_weights(
        self,
        reservoir_size: int | None = None,
        spectral_radius: float | None = None,
        connectivity: float | None = None,
        W_init: Callable[..., WeightsType] | None = None,
        sparsity_type: SparsityType = "csr",
        seed: int | None = None,
    ) -> WeightsType:
        if reservoir_size is None:
            reservoir_size = self._builder.reservoir_size
        if spectral_radius is None:
            spectral_radius = self._builder.spectral_radius
        if connectivity is None:
            connectivity = self._builder.connectivity
        if W_init is None:
            W_init = self._builder.W_init
        if seed is None:
            seed = self._builder.seed

        self._W = initialize_weights(
            (reservoir_size, reservoir_size),
            W_init,
            spectral_radius=spectral_radius,
            connectivity=connectivity,
            sparsity_type=sparsity_type,
            seed=seed,
        )
        return self._W

    def _initialize_bias(
        self,
        reservoir_size: int,
        input_connectivity: float,
        bias_scaling: float | bool,
        Win_init: Callable[..., WeightsType],
        seed: int | None,
    ) -> np.ndarray:
        bias = np.ravel(
            initialize_weights(
                (reservoir_size, 1),
                Win_init,
                scaling=bias_scaling,
                connectivity=input_connectivity,
                sparsity_type="dense",
                seed=seed,
            )
        )
        return bias

    def initialize_input_weights(
        self,
        input_dim: int,
        reservoir_size: int | None = None,
        input_scaling: float | Iterable[float] | None = None,
        input_connectivity: float | None = None,
        bias_scaling: float | bool | None = None,
        Win_init: Callable[..., WeightsType] | None = None,
        sparsity_type: SparsityType = "dense",
        seed: int | None = None,
    ) -> WeightsType:
        if reservoir_size is None:
            reservoir_size = self._builder.reservoir_size
        if input_scaling is None:
            input_scaling = self._builder.input_scaling
        if input_connectivity is None:
            input_connectivity = self._builder.input_connectivity
        if bias_scaling is None:
            bias_scaling = self._builder.bias_scaling
        elif bias_scaling is False:
            bias_scaling = 0.0
        if isinstance(bias_scaling, bool):
            bias_scaling = float(bias_scaling)
        if Win_init is None:
            Win_init = self._builder.Win_init
        if seed is None:
            seed = self._builder.seed

        # Calculate bias vector.
        self._bias = self._initialize_bias(reservoir_size, input_connectivity, bias_scaling, Win_init, seed)

        # Check if the given input scaling vector has correct number
        # of elements.
        if isinstance(input_scaling, Iterable):
            if len(list(input_scaling)) != input_dim:
                raise ValueError(f"The size of `input_scaling` is mismatched with `input_dim`.")

        # Calculate Win matrix.
        self._Win = initialize_weights(
            (reservoir_size, input_dim),
            Win_init,
            scaling=input_scaling,
            connectivity=input_connectivity,
            sparsity_type=sparsity_type,
            seed=seed,
        )
        return self._Win

    def kernel(self, u: np.ndarray, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        W = self.W
        Win = self.Win
        bias = self.bias

        g_in = self.noise_gain_in
        noise = self.noise_generator

        pre_x = Win @ (u + noise(shape=u.shape, gain=g_in)) + W @ x + bias

        # if self.has_feedback:
        #     Wfb = self.Wfb
        #     g_fb = self.noise_gain_fb
        #     h = self.fb_activation
        #     pre_y = self.feedback().reshape(-1, 1)
        #     y = h(pre_y) + noise(dist=dist, shape=pre_y.shape, gain=g_fb)

        #     pre_x += Wfb @ y

        return pre_x
