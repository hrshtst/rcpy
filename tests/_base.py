# ruff: noqa: ERA001
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


def gen_rng(seed: int = 0) -> np.random.RandomState:
    return np.random.RandomState(seed)


class DataSetBase:
    X: Iterable
    Y: Iterable

    def __iter__(self):
        for x, y in zip(self.X, self.Y, strict=False):
            yield x, y


class RandDataSet(DataSetBase):
    def __init__(self, n_samples: int = 10, n_features: int = 5, n_output: int = 1, seed: int = 0) -> None:
        self.rng = gen_rng(seed)
        self.X = self.rng.randn(n_samples, n_features)
        self.Y = self.rng.randn(n_samples, n_output)


class SmallDataSet(DataSetBase):
    def __init__(self) -> None:
        self.X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        #  y = 1 * x_0 + 2 * x_1
        self.Y = np.ravel(np.dot(self.X, self.coef.T))

    @property
    def coef(self) -> np.ndarray:
        return np.array([[1, 2]], dtype=float)


class SmallDataSetMO(DataSetBase):
    def __init__(self) -> None:
        self.X = np.array([[1, 1, 1], [1, 2, 1], [2, 2, 2], [2, 3, 2], [3, 2, 2]])
        #  y0 = 1 * x_0 +  2 * x_1 + -2 * x_2
        #  y1 = 2 * x_0 + -1 * x_1 +  1 * x_2
        self.Y = np.dot(self.X, self.coef.T)

    @property
    def coef(self) -> np.ndarray:
        return np.array([[1, 2, -2], [2, -1, 1]], dtype=float)


class LargeDataSet(DataSetBase):
    def __init__(self, n_samples: int = 500, seed: int = 0) -> None:
        self.rng = gen_rng(seed)
        self.X = self.rng.normal(0, 1, (n_samples, 4))
        self.noise = self.rng.normal(0, 0.1, n_samples)
        self.Y = self._gen_target(self.X, self.noise)

    def _gen_target(self, X: np.ndarray, noise: np.ndarray) -> np.ndarray:
        # Y = 2 * X[:, 0] + 0.1 * X[:, 1] - 4 * X[:, 2] + 0.5 * X[:, 3] + noise
        Y = np.ravel(np.dot(X, self.coef.T)) + noise
        _ = Y
        return Y

    @property
    def coef(self) -> np.ndarray:
        return np.array([[2.0, 0.1, -4.0, 0.5]])
