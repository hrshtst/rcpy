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
    input_bias: bool = True
    bias_scaling: float = 1.0
    seed: int | None = None


class Reservoir(object):
    _builder: ReservoirBuilder
    _W: WeightsType
    _Win: WeightsType
    _has_input_bias: bool
    _leaking_rate: float
    _seed: int | None

    def __init__(self, builder: ReservoirBuilder):
        self._builder = copy.copy(builder)
        self._leaking_rate = self._builder.leaking_rate
        self._seed = self._builder.seed
        self.initialize_internal_weights()

    @property
    def W(self) -> WeightsType:
        return self._W

    @property
    def Win(self) -> WeightsType:
        return self._Win

    @property
    def has_input_bias(self) -> bool:
        return self._has_input_bias

    @property
    def leaking_rate(self) -> float:
        return self._leaking_rate

    @property
    def size(self) -> int:
        return self._W.shape[0]

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

    def initialize_input_weights(
        self,
        input_dim: int,
        reservoir_size: int | None = None,
        input_scaling: float | Iterable[float] | None = None,
        input_connectivity: float | None = None,
        input_bias: bool | None = None,
        bias_scaling: float | None = None,
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
        if input_bias is None:
            input_bias = self._builder.input_bias
        if bias_scaling is None:
            bias_scaling = self._builder.bias_scaling
        if Win_init is None:
            Win_init = self._builder.Win_init
        if seed is None:
            seed = self._builder.seed

        # Check if the given input scaling vector has correct number
        # of elements.
        if isinstance(input_scaling, Iterable):
            if len(list(input_scaling)) != input_dim:
                raise ValueError(f"The size of `input_scaling` is mismatched with `input_dim`.")

        # Calculate scaling vector.
        self._has_input_bias = input_bias
        if self.has_input_bias:
            if isinstance(input_scaling, float):
                scaling = [bias_scaling, input_scaling]
            else:
                scaling = np.insert(np.array(input_scaling), 0, bias_scaling)
        else:
            if isinstance(input_scaling, float):
                scaling = input_scaling
            else:
                scaling = np.array(input_scaling)

        self._Win = initialize_weights(
            (reservoir_size, input_dim + (1 if input_bias else 0)),
            Win_init,
            scaling=scaling,
            connectivity=input_connectivity,
            sparsity_type=sparsity_type,
            seed=seed,
        )
        return self._Win
