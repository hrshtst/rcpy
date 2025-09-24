# src/rcpy/lms.py
import numpy as np
import taichi as ti


@ti.data_oriented
class TaichiLMS:
    """A Least Mean Squares (LMS) solver using Taichi."""

    def __init__(self, n_reservoir, n_output, learning_rate=0.1):
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.W_out = ti.field(dtype=ti.f32, shape=(n_output, n_reservoir))

        self.x_ti = ti.field(dtype=ti.f32, shape=self.n_reservoir)
        self.y_target_ti = ti.Vector.field(self.n_output, dtype=ti.f32, shape=())

    @ti.kernel
    def _update_kernel(self, x: ti.template(), y_target: ti.template()):
        # Compute prediction error
        y_pred = ti.Vector([0.0 for _ in range(self.n_output)])
        for i, j in self.W_out:
            y_pred[i] += self.W_out[i, j] * x[j]
        e = y_target[None] - y_pred

        # Update output weights
        for i in range(self.n_output):
            for j in range(self.n_reservoir):
                self.W_out[i, j] += self.learning_rate * e[i] * x[j]

    def update(self, x_np, y_target_np):
        """Performs a single step of LMS update."""
        self.x_ti.from_numpy(x_np)
        self.y_target_ti.from_numpy(y_target_np)
        self._update_kernel(self.x_ti, self.y_target_ti)

    def fit(self, X_np, Y_np):
        print("  Solving for W_out using Taichi LMS...")
        n_samples = X_np.shape[1]

        for t in range(n_samples):
            self.update(X_np[:, t], Y_np[:, t])

        return self.W_out.to_numpy()


class NumpyLMS:
    """A Least Mean Squares (LMS) solver using NumPy."""

    def __init__(self, n_reservoir, n_output, learning_rate=0.1):
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.W_out = np.zeros((n_output, n_reservoir), dtype=np.float32)

    def update(self, x_t, y_target_t):
        """Performs a single step of LMS update."""
        # Compute prediction error
        y_pred_t = self.W_out @ x_t
        e_t = y_target_t - y_pred_t

        # Update output weights
        self.W_out += self.learning_rate * np.outer(e_t, x_t)

    def fit(self, X, Y):
        print("  Solving for W_out using NumPy LMS...")
        n_samples = X.shape[1]

        for t in range(n_samples):
            self.update(X[:, t], Y[:, t])

        return self.W_out
