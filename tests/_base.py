from typing import Iterable

import numpy as np


def gen_rng(seed: int = 0) -> np.random.RandomState:
    return np.random.RandomState(seed)


class DataSetBase:
    X: Iterable
    Y: Iterable

    def __iter__(self):
        for x, y in zip(self.X, self.Y):
            yield x, y


class RandDataSet(DataSetBase):
    def __init__(self, n_samples: int = 10, n_features: int = 5, n_output: int = 1, seed: int = 0):
        self.rng = gen_rng(seed)
        self.X = self.rng.randn(n_samples, n_features)
        self.Y = self.rng.randn(n_samples, n_output)


class SmallDataSet(DataSetBase):
    def __init__(self):
        self.X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        #  y = 1 * x_0 + 2 * x_1
        self.Y = np.dot(self.X, np.array([1, 2]))

    @property
    def coef(self):
        return np.array([[1.0, 2.0]])
