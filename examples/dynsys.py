#!/usr/bin/env python

from __future__ import annotations

import copy
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from rcpy.random import get_rng, set_seed
from rcpy.reservoir import Reservoir, ReservoirBuilder
from rcpy.ridge import Ridge
from rcpy.weights import zeros

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from rcpy._type import Unknown
    from rcpy.readout import Readout


def lorenz(
    _: float,
    xyz: tuple[float, float, float],
    sigma: float,
    beta: float,
    rho: float,
) -> tuple[float, float, float]:
    x, y, z = xyz
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return (dx, dy, dz)


def vdp(_: float, uv: tuple[float, float], mu: float) -> tuple[float, float]:
    u, v = uv
    du = v
    dv = mu * (1.0 - u * u) * v - u
    return (du, dv)


def lq(_: float, state: tuple[float, float], a: float, b: float, c: float) -> tuple[float, float]:
    y, dy = state
    ddy = -(b * dy + c * y) / a
    return (dy, ddy)


class DynSys:
    def __init__(
        self,
        f: Callable,
        *params: float,
        y0_low: float | None = None,
        y0_high: float | None = None,
    ) -> None:
        self.f = f
        self.params = params
        self.dim = self.estimate_dim(f)
        self.y0_low = y0_low
        self.y0_high = y0_high
        self.rng = get_rng(12345)  # Get new RNG instance to generate consistent initial values.

    @staticmethod
    def estimate_dim(f: Callable) -> int:
        n_params = len(signature(f).parameters) - 2
        dummy_params = [1.0 for _ in range(n_params)]

        dim = 1
        found = False
        while not found:
            y = np.zeros((dim,))
            try:
                f(0.0, y, *dummy_params)
            except ValueError:
                dim += 1
            else:
                found = True
        return dim

    def _y0_lim(self, y0_low: float | None, y0_high: float | None) -> tuple[float, float]:
        if y0_low is None:
            y0_low = self.y0_low if self.y0_low is not None else 0.0
        if y0_high is None:
            y0_high = self.y0_high if self.y0_high is not None else 1.0
        return y0_low, y0_high

    def solve(
        self,
        T: float,
        dt: float = 1.0,
        y0: Iterable | None = None,
        y0_low: float | None = None,
        y0_high: float | None = None,
    ) -> Unknown:
        if y0 is None:
            y0_low, y0_high = self._y0_lim(y0_low, y0_high)
            y0 = self.rng.uniform(y0_low, y0_high, (self.dim,))
        else:
            y0 = np.array(y0, dtype=float)
        t_eval = np.arange(0.0, T, dt)
        solver = solve_ivp(self.f, (0.0, T), y0, t_eval=t_eval, args=self.params)
        return solver.y.T

    def __call__(
        self,
        T: float,
        dt: float = 1.0,
        y0: Iterable | None = None,
        y0_low: float | None = None,
        y0_high: float | None = None,
    ) -> Unknown:
        return self.solve(T, dt, y0, y0_low, y0_high)


@dataclass
class EsnBuilder:
    reservoir: ReservoirBuilder
    enable_feedback: bool = False
    teacher_forcing: bool = True
    warmup: int = 0


class Esn:
    _builder: EsnBuilder
    _enable_feedback: bool
    _teacher_forcing: bool
    _warmup: int
    _reservoir: Reservoir
    _readout: Readout
    _input_dim: int
    _output_dim: int

    def __init__(self, builder: EsnBuilder) -> None:
        self._builder = copy.copy(builder)
        self._enable_feedback = self._builder.enable_feedback
        self._teacher_forcing = self._builder.teacher_forcing
        self._warmup = self._builder.warmup
        self._reservoir = Reservoir(self._builder.reservoir)
        self._readout = Ridge(1e-4)
        self._readout.batch_interval = 1

    @property
    def y0_default(self) -> np.ndarray:
        return zeros(self._output_dim)

    @property
    def reservoir(self) -> Reservoir:
        return self._reservoir

    @property
    def readout(self) -> Readout:
        return self._readout

    def fit(
        self,
        U: np.ndarray,
        D: np.ndarray,
        *,
        y0: np.ndarray | None = None,
        enable_feedback: bool | None = None,
        teacher_forcing: bool | None = None,
        warmup: int | None = None,
    ) -> Esn:
        if enable_feedback is None:
            enable_feedback = self._builder.enable_feedback
        if teacher_forcing is None:
            teacher_forcing = self._builder.teacher_forcing
        if warmup is None:
            warmup = self._builder.warmup

        if D.ndim == 1:
            D = np.atleast_2d(D).T
        self._input_dim = U.shape[1]
        self._output_dim = D.shape[1]
        self._reservoir.initialize_input_weights(self._input_dim)
        if enable_feedback:
            self._reservoir.initialize_feedback_weights(self._output_dim)

        y = y0 if y0 is not None else self.y0_default
        for i, (u, d) in enumerate(zip(U, D, strict=True)):
            if self._reservoir.has_feedback():
                x = self._reservoir.forward(u, y)
            else:
                x = self._reservoir.forward(u)
            if warmup < i:
                self._readout.backward(x, d)
            y = d if teacher_forcing else self._readout.predict(x)
        # self._readout.fit() # noqa: ERA001
        return self

    def forward(self, u: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        x = self._reservoir.forward(u, y)
        y_next = self._readout.predict(x)
        return y_next

    def predict(self, U: np.ndarray, y0: np.ndarray | None = None) -> np.ndarray:
        y_pred: list[np.ndarray] = []
        y = y0 if y0 is not None else self.y0_default
        for u in U:
            if self._reservoir.has_feedback():
                y_next = self.forward(u, y)
            else:
                y_next = self.forward(u)
            y_pred.append(y_next)
            y = y_next
        return np.asarray(y_pred)

    def run(self, U: np.ndarray, y0: np.ndarray | None = None) -> np.ndarray:
        y_pred: list[np.ndarray] = []
        y = y0 if y0 is not None else self.y0_default
        for u in U:
            y_next = self.forward(u, y)
            y_pred.append(y_next)
            y = y_next
        return np.asarray(y_pred)

    def run2(self, U: np.ndarray) -> np.ndarray:
        y_pred: list[np.ndarray] = []
        y = U[0]
        for _ in range(len(U)):
            y_next = self.forward(y)
            y_pred.append(y_next)
            y = y_next
        return np.asarray(y_pred)


def plot2d(
    all_data: Iterable[np.ndarray],
    title: str | None = None,
    filename: str | None = None,
) -> tuple[Figure, Axes]:
    fig = plt.figure(figsize=(7, 7))
    ax: Axes = fig.add_subplot(111)
    for data in all_data:
        (line,) = ax.plot(data[:, 0][0], data[:, 1][0], marker="x", ms=None)
        ax.plot(data[:, 0], data[:, 1], c=line.get_color())
    plt.xlabel("x")
    plt.ylabel("y")
    if title is not None:
        plt.title(title)
    ax.set_aspect("equal", "box")
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    return fig, ax


def test_vdp(
    T: float = 50.0,
    N_data: int = 10,
    N_test: int = 10,
    dt: float = 0.01,
    N_x: int = 300,
    density: float = 0.1,
    rho: float = 0.95,
    lr: float = 0.99,
    input_scale: float = 0.1,
) -> None:
    # training section
    n = int(T / dt)
    ds = DynSys(vdp, 1.0, y0_low=-5, y0_high=5)
    train_dataset = [ds(T + dt, dt) for _ in range(N_data)]

    res = ReservoirBuilder(
        N_x,
        rho,
        density,
        lr,
        input_scaling=input_scale,
        fb_scaling=input_scale,
        bias_scaling=0.0,
        noise_gain_in=0.0,
    )
    model = Esn(EsnBuilder(res, enable_feedback=True))

    print("Training in progress...")  # noqa: T201
    for data in train_dataset:
        U = np.zeros((n, 0))  # No input
        y0 = data[0]
        D = data[1:]
        model.reservoir.reset_reservoir_state()
        model.fit(U, D, y0=y0)

    # test section
    test_dataset = [ds(T, dt) for _ in range(N_test)]
    pred_dataset = []
    for i, data in enumerate(test_dataset):
        print(f"Testing trajectory starting from {data[0]} ({i + 1}/{N_test})...")  # noqa: T201
        model.reservoir.reset_reservoir_state()
        U = np.zeros((n, 0))  # No input
        Y = model.run(U, y0=data[0])
        Y = np.vstack((data[0], Y))  # insert initial value
        pred_dataset.append(Y)

    # plot section
    plt.rcParams["figure.dpi"] = 75
    plot2d(pred_dataset, title="Autonomous running of ESN")
    plot2d(train_dataset, title="Trajectories used for training")
    plot2d(test_dataset, title="Trajectories of VdP for reference")


def test_vdp2(
    T: float = 50.0,
    N_data: int = 10,
    N_test: int = 10,
    dt: float = 0.01,
    N_x: int = 300,
    density: float = 0.1,
    rho: float = 0.95,
    lr: float = 0.99,
    input_scale: float = 0.1,
) -> None:
    # training section
    int(T / dt)
    ds = DynSys(vdp, 1.0, y0_low=-5, y0_high=5)
    train_dataset = [ds(T + dt, dt) for _ in range(N_data)]

    res = ReservoirBuilder(
        N_x,
        rho,
        density,
        lr,
        input_scaling=input_scale,
        fb_scaling=input_scale,
        bias_scaling=0.0,
        noise_gain_in=0.0,
    )
    model = Esn(EsnBuilder(res))

    print("Training in progress...")  # noqa: T201
    for i, data in enumerate(train_dataset):
        print(f"Training data ({i + 1}/{N_data})...")  # noqa: T201
        U, D = data[:-1], data[1:]
        model.reservoir.reset_reservoir_state()
        model.fit(U, D)
    model.readout.fit()

    # test section
    test_dataset = [ds(T, dt) for _ in range(N_test)]
    pred_dataset = []
    for i, data in enumerate(test_dataset):
        print(f"Testing trajectory starting from {data[0]} ({i + 1}/{N_test})...")  # noqa: T201
        model.reservoir.reset_reservoir_state()
        y_pred = model.run2(data)
        y_pred = np.vstack((data[0], y_pred))  # insert initial value
        pred_dataset.append(y_pred)

    # plot section
    plt.rcParams["figure.dpi"] = 75
    plot2d(pred_dataset, title="Autonomous running of ESN")
    plot2d(train_dataset, title="Trajectories used for training")
    plot2d(test_dataset, title="Trajectories of VdP for reference")


def main() -> None:
    set_seed(12345)
    test_vdp2()
    plt.show()


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "VdP env lr noqa rc usr"
# End:
