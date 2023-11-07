import numpy as np

from rebasicspy.readout import Readout


class Ridge(Readout):
    _regularization: float
    _X: np.ndarray
    _Y: np.ndarray
    _XXT: np.ndarray
    _YXT: np.ndarray
    _sample_weight: np.ndarray

    def __init__(self, regularization: float | None = None, batch_interval: int | None = None):
        if regularization is None:
            self.regularization = 0.0
        else:
            self.regularization = regularization
        super().__init__(batch_interval=batch_interval)

    @property
    def regularization(self) -> float:
        return self._regularization

    @regularization.setter
    def regularization(self, beta: float) -> float:
        self._regularization = beta
        return self._regularization

    def stack(self, x: np.ndarray, y_target: float | int | np.ndarray, sample_weight: float | int | None):
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

    def accumulate(self):
        try:
            rooted_sample_weight = np.sqrt(self._sample_weight).reshape(-1, 1)
        except AttributeError:
            rooted_sample_weight = np.ones((len(self._X), 1))
        sX = rooted_sample_weight * self._X
        sY = rooted_sample_weight * self._Y

        try:
            self._XXT += np.dot(sX.T, sX)
            self._YXT += np.dot(sY.T, sX)
        except AttributeError:
            self._XXT = np.dot(sX.T, sX)
            self._YXT = np.dot(sY.T, sX)

    def solve(self) -> np.ndarray:
        X_pseudo_inv = np.linalg.inv(self._XXT + self.regularization * np.identity(len(self._XXT), dtype=float))
        return np.dot(self._YXT, X_pseudo_inv)

    def process_backward(
        self,
        x: np.ndarray,
        y_target: float | int | np.ndarray,
        sample_weight: float | int | None,
    ):
        self.stack(x, y_target, sample_weight)
        return super().process_backward(x, y_target, sample_weight)

    def process_backward_batch(self):
        self.accumulate()
        del self._X
        del self._Y
        if hasattr(self, "_sample_weight"):
            del self._sample_weight
        return super().process_backward_batch()

    def finalize_backward_batch(self):
        if self._batch_count > 0:
            self.process_backward_batch()
        return super().finalize_backward_batch()

    def reset(self):
        for attr in ("_X", "_Y", "_XXT", "_YXT", "_sample_weight"):
            self.__dict__.pop(attr, None)
        return super().reset()

    def fit(self):
        self.finalize_backward_batch()
        self._Wout = self.solve()
        return super().fit()
