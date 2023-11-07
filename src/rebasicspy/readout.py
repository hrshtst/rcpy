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
        if hasattr(self, "_batch_processed") and not self._batch_processed:
            self.finalize_backward_batch()
        self._batch_count = 0
        self._batch_processed = False
        self._batch_finalized = False

    def fit(self):
        pass
