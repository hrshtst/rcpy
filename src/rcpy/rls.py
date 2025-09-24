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

    @ti.kernel
    def _update_kernel(self, x: ti.template(), y_target: ti.template()):
        # Compute prediction error
        y_pred = ti.Vector([0.0 for _ in range(self.n_output)])
        for i, j in self.W_out:
            y_pred[i] += self.W_out[i, j] * x[j]
        e = y_target - y_pred

        # Compute gain vector
        Px = ti.Vector([0.0 for _ in range(self.n_reservoir)])
        for i, j in self.P:
            Px[i] += self.P[i, j] * x[j]

        k_numerator = Px
        k_denominator = self.forgetting_factor + x.dot(Px)
        k = k_numerator / k_denominator

        # Update output weights
        for i in range(self.n_output):
            for j in range(self.n_reservoir):
                self.W_out[i, j] += e[i] * k[j]

        # Update inverse correlation matrix
        k_outer_Px = ti.Matrix([[k[i] * Px[j] for j in range(self.n_reservoir)] for i in range(self.n_reservoir)])
        P_new = (1.0 / self.forgetting_factor) * (self.P - k_outer_Px)

        for i, j in self.P:
            self.P[i, j] = P_new[i, j]

    def fit(self, X_np, Y_np):
        print("  Solving for W_out using Taichi RLS...")
        n_samples = X_np.shape[1]

        x_ti = ti.field(dtype=ti.f32, shape=self.n_reservoir)
        y_target_ti = ti.field(dtype=ti.f32, shape=self.n_output)

        for t in range(n_samples):
            x_ti.from_numpy(X_np[:, t])
            y_target_ti.from_numpy(Y_np[:, t])
            self._update_kernel(x_ti, y_target_ti)

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

    def fit(self, X, Y):
        print("  Solving for W_out using NumPy RLS...")
        n_samples = X.shape[1]

        for t in range(n_samples):
            x_t = X[:, t]
            y_target_t = Y[:, t]

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

        return self.W_out
