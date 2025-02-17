from __future__ import annotations

import numpy as np

from rcpy.random import get_rng

_ERR_MSG_READOUT_OPT_NOT_STARTED = "Optimization of readout layer has not started yet."


class Readout:
    _Wout: np.ndarray
    _batch_interval: int
    _batch_count: int
    _batch_processed: bool
    _batch_finalized: bool

    def __init__(self, batch_interval: int | None = None) -> None:
        self.batch_interval = batch_interval  # type: ignore[assignment]
        self.reset()

    @property
    def batch_interval(self) -> int:
        return self._batch_interval

    @batch_interval.setter
    def batch_interval(self, val: int | None) -> None:
        if val is None:
            self._batch_interval = 0
        else:
            self._batch_interval = val

    def has_unprocessed_batch(self) -> bool:
        return self._batch_count > 0

    def has_nothing_to_process(self) -> bool:
        return self._batch_count == 0 and not self._batch_processed

    @property
    def Wout(self) -> np.ndarray:
        if not hasattr(self, "_Wout"):
            raise RuntimeError(_ERR_MSG_READOUT_OPT_NOT_STARTED)
        return self._Wout

    def backward(self, x: np.ndarray, y_target: float | np.ndarray, sample_weight: float | None = None) -> None:
        self._batch_count += 1
        self._batch_processed = False
        self._batch_finalized = False
        self.process_backward(x, y_target, sample_weight)
        if self._batch_count == self.batch_interval:
            self.process_backward_batch()

    def process_backward(self, x: np.ndarray, y_target: float | np.ndarray, sample_weight: float | None) -> None:
        pass

    def process_backward_batch(self) -> None:
        self._batch_count = 0
        self._batch_processed = True

    def finalize_backward_batch(self) -> None:
        self._batch_count = 0
        self._batch_processed = True
        self._batch_finalized = True

    def reset(self) -> None:
        self._batch_count = 0
        self._batch_processed = False
        self._batch_finalized = False
        if hasattr(self, "_Wout"):
            del self._Wout

    def finalize(self) -> None:
        # Finalize Wout here in derived class
        raise NotImplementedError

    def fit(self) -> None:
        if self.has_nothing_to_process():
            raise RuntimeError(_ERR_MSG_READOUT_OPT_NOT_STARTED)

        if self.has_unprocessed_batch():
            self.finalize_backward_batch()

        self.finalize()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.Wout @ x


def rescale_data(
    X: np.ndarray,
    y: float | np.ndarray,
    sample_weight: float | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_weight is None:
        return X, np.atleast_1d(y)

    sample_weight_sqrt = np.sqrt(np.atleast_1d(sample_weight))
    if X.ndim == 1:
        X = X * sample_weight_sqrt
    else:
        X = X * sample_weight_sqrt[:, np.newaxis]

    y = np.atleast_1d(y)
    if y.ndim == 1:
        y = y * sample_weight_sqrt
    else:
        y = y * sample_weight_sqrt[:, np.newaxis]
    return X, y  # type: ignore[return-value]


def initialize_weights(shape: tuple[int, ...], initializer: str = "zeros") -> np.ndarray:
    if initializer == "random":
        return get_rng().normal(0.0, 0.5, size=shape)
    if initializer == "zeros":
        return np.zeros(shape, dtype=float)
    msg = f"Unknown initializer: {initializer}"
    raise ValueError(msg)


def compute_error(readout: Readout, x: np.ndarray, y_target: float | np.ndarray) -> np.ndarray:
    y = readout.predict(x)
    return y_target - y  # return error


# Local Variables:
# jinx-local-words: "Wout"
# End:
