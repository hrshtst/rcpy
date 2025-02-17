from __future__ import annotations

import numpy as np
from scipy import linalg

from rcpy.readout import Readout, rescale_data


class Ridge(Readout):
    _regularization: float
    _X: np.ndarray
    _Y: np.ndarray
    _XXT: np.ndarray
    _YXT: np.ndarray
    _sample_weight: np.ndarray
    _solver: str

    def __init__(
        self,
        regularization: float | None = None,
        solver: str = "cholesky",
        batch_interval: int | None = None,
    ) -> None:
        if regularization is None:
            self.regularization = 0.0
        else:
            self.regularization = regularization
        self._solver = solver
        super().__init__(batch_interval=batch_interval)

    @property
    def regularization(self) -> float:
        return self._regularization

    @regularization.setter
    def regularization(self, beta: float) -> float:
        self._regularization = beta
        return self._regularization

    @property
    def solver(self) -> str:
        return self._solver

    @solver.setter
    def solver(self, _solver: str) -> str:
        self._solver = _solver
        return self._solver

    def stack(self, x: np.ndarray, y_target: float | np.ndarray, sample_weight: float | None) -> None:
        try:
            self._X = np.vstack((self._X, np.reshape(x, (1, -1))))
            self._Y = np.vstack((self._Y, np.reshape(y_target, (1, -1))))
        except AttributeError:
            self._X = np.reshape(x, (1, -1)).astype(float)
            self._Y = np.reshape(y_target, (1, -1)).astype(float)

        if sample_weight is not None:
            try:
                self._sample_weight = np.concatenate((self._sample_weight, np.asarray([sample_weight])))
            except AttributeError:
                self._sample_weight = np.asarray([sample_weight], dtype=float)

    def accumulate(self) -> None:
        try:
            X, Y = rescale_data(self._X, self._Y, self._sample_weight)
        except AttributeError:
            X, Y = self._X, self._Y

        try:
            self._XXT += np.dot(X.T, X)
            self._YXT += np.dot(Y.T, X)
        except AttributeError:
            self._XXT = np.dot(X.T, X)
            self._YXT = np.dot(Y.T, X)

    @staticmethod
    def solve(solver: str, XXT: np.ndarray, YXT: np.ndarray, alpha: float) -> np.ndarray:
        ridge = alpha * np.identity(len(XXT), dtype=float)
        if solver == "pseudoinv":
            X_pseudo_inv = np.linalg.inv(XXT + ridge)
            return np.dot(YXT, X_pseudo_inv)
        if solver == "cholesky":
            w = linalg.solve(XXT + ridge, YXT.T, assume_a="sym")
            return w.T
        msg = f"Unknown solver: {solver}. Choose from 'pseudoinv' or 'cholesky'."
        raise ValueError(msg)

    def process_backward(
        self,
        x: np.ndarray,
        y_target: float | np.ndarray,
        sample_weight: float | None,
    ) -> None:
        self.stack(x, y_target, sample_weight)
        return super().process_backward(x, y_target, sample_weight)

    def process_backward_batch(self) -> None:
        self.accumulate()
        del self._X
        del self._Y
        if hasattr(self, "_sample_weight"):
            del self._sample_weight
        return super().process_backward_batch()

    def finalize_backward_batch(self) -> None:
        if self._batch_count > 0:
            self.process_backward_batch()
        return super().finalize_backward_batch()

    def reset(self) -> None:
        for attr in ("_X", "_Y", "_XXT", "_YXT", "_sample_weight"):
            self.__dict__.pop(attr, None)
        return super().reset()

    def finalize(self) -> None:
        self._Wout = self.solve(self._solver, self._XXT, self._YXT, self.regularization)


# Local Variables:
# jinx-local-words: "cholesky pseudoinv sym"
# End:
