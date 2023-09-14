import copy
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from rebasicspy._type import SparsityType, WeightsType
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
    _seed: int | None

    def __init__(self, builder: ReservoirBuilder):
        self._builder = copy.copy(builder)
        self._leaking_rate = self._builder.leaking_rate
        self._seed = self._builder.seed
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
