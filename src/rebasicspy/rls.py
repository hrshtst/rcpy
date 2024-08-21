from __future__ import annotations

import warnings

import numpy as np

from rebasicspy.readout import Readout, compute_error, initialize_weights


class RLS(Readout):
    _forgetting_factor: float
    _delta: float
    _Wout_init: str
    _P: np.ndarray

    def __init__(self, forgetting_factor: float = 0.98, delta: float = 0.001, Wout_init: str = "zeros") -> None:
        self.forgetting_factor = forgetting_factor
        self.delta = delta
        self._Wout_init = Wout_init
        super().__init__(batch_interval=1)

    @property
    def forgetting_factor(self) -> float:
        return self._forgetting_factor

    @forgetting_factor.setter
    def forgetting_factor(self, lr: float) -> float:
        self._forgetting_factor = lr
        return self._forgetting_factor

    @property
    def delta(self) -> float:
        return self._delta

    @delta.setter
    def delta(self, delta: float) -> float:
        self._delta = delta
        return self._delta

    def process_backward(self, x: np.ndarray, y_target: float | np.ndarray, sample_weight: float | None) -> None:
        # Error estimation
        try:
            e = compute_error(self, x, y_target)
        except RuntimeError:
            # Initialize Wout and inverse of autocorrelation matrix
            y_target = np.atleast_1d(y_target)
            self._Wout = initialize_weights((len(y_target), len(x)), initializer=self._Wout_init)
            self._P = (1.0 / self.delta) * np.identity(len(x))
            # Warning if sample_weight is given
            if sample_weight is not None:
                msg = "Least-squares method does not support weighted algorithms"
                warnings.warn(msg, UserWarning, stacklevel=2)
            e = compute_error(self, x, y_target)

        # Computation of gain vector
        u = np.dot(self._P, x)
        k = (1.0 / (self.forgetting_factor + np.dot(x, u))) * u

        # Inverse of autocorrelation matrix update
        self._P = (1.0 / self.forgetting_factor) * (self._P - np.outer(k, u))
        # Make the inverse of autocorrelation matrix symmetric to
        # avoid numerical instability.
        tril = np.tril_indices(self._P.shape[0])
        self._P[tril] = self._P.T[tril]

        # Tap-weight matrix adaptation
        dw = np.outer(e, k)
        self._Wout += dw
        return super().process_backward(x, y_target, sample_weight)

    def process_backward_batch(self) -> None:
        # Do nothing
        return super().process_backward_batch()

    def finalize_backward_batch(self) -> None:
        # Do nothing
        return super().finalize_backward_batch()

    def finalize(self) -> None:
        # Do nothing
        pass

    def reset(self) -> None:
        return super().reset()


# Local Variables:
# jinx-local-words: "Wout autocorrelation"
# End:
