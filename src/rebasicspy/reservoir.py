import copy
from dataclasses import dataclass
from typing import Callable, Iterable

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
    add_bias: bool = True
    bias_scaling: float = 1.0
    seed: int | None = None


class Reservoir(object):
    _builder: ReservoirBuilder
    _W: WeightsType
    _Win: WeightsType
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
    def leaking_rate(self) -> float:
        return self._leaking_rate

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
