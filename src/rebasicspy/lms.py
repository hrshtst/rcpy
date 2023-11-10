import numpy as np

from rebasicspy.readout import Readout, compute_error, initialize_weights, rescale_data


class LMS(Readout):
    _learning_rate: float
    _Wout_init: str

    def __init__(self, learning_rate: float = 0.1, Wout_init: str = "zeros"):
        self.learning_rate = learning_rate
        self._Wout_init = Wout_init
        super().__init__(batch_interval=1)

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, lr: float) -> float:
        self._learning_rate = lr
        return self._learning_rate

    def process_backward(self, x: np.ndarray, y_target: float | int | np.ndarray, sample_weight: float | int | None):
        # Rescale data
        x, y_target = rescale_data(x, y_target, sample_weight)

        # Error estimation
        try:
            e = compute_error(self, x, y_target)
        except RuntimeError:
            # Initialize Wout
            self._Wout = initialize_weights((len(y_target), len(x)), initializer=self._Wout_init)
            e = compute_error(self, x, y_target)

        # Tap-weight matrix adaptation
        dw = self.learning_rate * np.outer(e, x)
        self._Wout += dw
        return super().process_backward(x, y_target, sample_weight)

    def process_backward_batch(self):
        # Do nothing
        return super().process_backward_batch()

    def finalize_backward_batch(self):
        # Do nothing
        return super().finalize_backward_batch()

    def finalize(self):
        # Do nothing
        pass

    def reset(self):
        return super().reset()
