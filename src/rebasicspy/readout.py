import numpy as np

_ERR_MSG_READOUT_OPT_NOT_STARTED = "Optimization of readout layer has not started yet."


class Readout(object):
    _Wout: np.ndarray
    _batch_interval: int
    _batch_count: int
    _batch_processed: bool
    _batch_finalized: bool

    def __init__(self, batch_interval: int | None = None):
        self.batch_interval = batch_interval
        self.reset()

    @property
    def batch_interval(self) -> int:
        return self._batch_interval

    @batch_interval.setter
    def batch_interval(self, val: int | None):
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

    def backward(self, x: np.ndarray, y_target: float | int | np.ndarray, sample_weight: float | int | None = None):
        self._batch_count += 1
        self._batch_processed = False
        self._batch_finalized = False
        self.process_backward(x, y_target, sample_weight)
        if self._batch_count == self.batch_interval:
            self.process_backward_batch()

    def process_backward(self, x: np.ndarray, y_target: float | int | np.ndarray, sample_weight: float | int | None):
        _ = x, y_target, sample_weight

    def process_backward_batch(self):
        self._batch_count = 0
        self._batch_processed = True

    def finalize_backward_batch(self):
        self._batch_count = 0
        self._batch_processed = True
        self._batch_finalized = True

    def reset(self):
        self._batch_count = 0
        self._batch_processed = False
        self._batch_finalized = False
        if hasattr(self, "_Wout"):
            del self._Wout

    def finalize(self):
        # Finalize Wout here in derived class
        raise NotImplementedError

    def fit(self):
        if self.has_nothing_to_process():
            raise RuntimeError(_ERR_MSG_READOUT_OPT_NOT_STARTED)

        if self.has_unprocessed_batch():
            self.finalize_backward_batch()

        self.finalize()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.Wout @ x[:, np.newaxis]


def initialize_weights(shape: tuple[int, ...], initializer: str = "zeros") -> np.ndarray:
    if initializer == "random":
        return np.random.normal(0.0, 0.5, size=shape)
    elif initializer == "zeros":
        return np.zeros(shape, dtype=float)
    else:
        raise ValueError(f"Unknown initializer: {initializer}")
