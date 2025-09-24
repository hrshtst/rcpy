# src/rcpy/rls.py
import numpy as np
import taichi as ti


@ti.data_oriented
class TaichiRLS:
    """A Recursive Least Squares (RLS) solver using Taichi."""

    def __init__(self, n_reservoir, n_output, forgetting_factor=0.98, delta=0.001):
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.forgetting_factor = forgetting_factor
        self.delta = delta
        self.W_out = ti.field(dtype=ti.f32, shape=(n_output, n_reservoir))
        self.P = ti.field(dtype=ti.f32, shape=(n_reservoir, n_reservoir))
        self.P.from_numpy((1.0 / self.delta) * np.identity(n_reservoir, dtype=np.float32))

        self.x_ti = ti.field(dtype=ti.f32, shape=self.n_reservoir)
        self.y_target_ti = ti.Vector.field(self.n_output, dtype=ti.f32, shape=())

    @ti.kernel
    def _update_kernel(self, x: ti.template(), y_target: ti.template()):
        # Compute prediction error
        y_pred = ti.Vector([0.0 for _ in range(self.n_output)])
        for i, j in self.W_out:
            y_pred[i] += self.W_out[i, j] * x[j]
        e = y_target[None] - y_pred

        # Compute gain vector
        Px = ti.Vector([0.0 for _ in range(self.n_reservoir)])
        for i, j in self.P:
            Px[i] += self.P[i, j] * x[j]

        x_dot_Px = 0.0
        for i in range(self.n_reservoir):
            x_dot_Px += x[i] * Px[i]

        k_numerator = Px
        k_denominator = self.forgetting_factor + x_dot_Px
        k = k_numerator / k_denominator

        # Update output weights
        for i in range(self.n_output):
            for j in range(self.n_reservoir):
                self.W_out[i, j] += e[i] * k[j]

        # Update inverse correlation matrix
        k_outer_Px = ti.Matrix([[k[i] * Px[j] for j in range(self.n_reservoir)] for i in range(self.n_reservoir)])

        for i, j in self.P:
            self.P[i, j] = (1.0 / self.forgetting_factor) * (self.P[i, j] - k_outer_Px[i, j])

    def update(self, x_np, y_target_np):
        """Performs a single step of RLS update."""
        self.x_ti.from_numpy(x_np)
        self.y_target_ti.from_numpy(y_target_np)
        self._update_kernel(self.x_ti, self.y_target_ti)

    def fit(self, X_np, Y_np):
        print("  Solving for W_out using Taichi RLS...")
        n_samples = X_np.shape[1]

        for t in range(n_samples):
            self.update(X_np[:, t], Y_np[:, t])

        return self.W_out.to_numpy()


class NumpyRLS:
    """A Recursive Least Squares (RLS) solver using NumPy."""

    def __init__(self, n_reservoir, n_output, forgetting_factor=0.98, delta=0.001):
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.forgetting_factor = forgetting_factor
        self.delta = delta
        self.W_out = np.zeros((n_output, n_reservoir), dtype=np.float32)
        self.P = (1.0 / self.delta) * np.identity(n_reservoir, dtype=np.float32)

    def update(self, x_t, y_target_t):
        """Performs a single step of RLS update."""
        # Compute prediction error
        y_pred_t = self.W_out @ x_t
        e_t = y_target_t - y_pred_t

        # Compute gain vector
        Px = self.P @ x_t
        k_denominator = self.forgetting_factor + x_t.T @ Px
        k = Px / k_denominator

        # Update output weights
        self.W_out += np.outer(e_t, k)

        # Update inverse correlation matrix
        self.P = (1.0 / self.forgetting_factor) * (self.P - np.outer(k, Px))

    def fit(self, X, Y):
        print("  Solving for W_out using NumPy RLS...")
        n_samples = X.shape[1]

        for t in range(n_samples):
            self.update(X[:, t], Y[:, t])

        return self.W_out
