import numpy as np
import taichi as ti


@ti.data_oriented
class TaichiRidge:
    """A Ridge Regression solver using the Conjugate Gradient method in Taichi."""

    def __init__(self, alpha=1e-4, n_iter=30):
        self.alpha = alpha
        self.n_iter = n_iter
        self.A, self.x_vec, self.r_vec, self.p_vec, self.Ap_vec = None, None, None, None, None

    @ti.kernel
    def _compute_A(self, X: ti.template()):
        n_res, n_samples = X.shape
        # This is equivalent to A = X @ X.T
        for i, j in self.A:
            sum_val = 0.0
            for k in range(n_samples):
                sum_val += X[i, k] * X[j, k]
            self.A[i, j] = sum_val
        for i in range(n_res):
            self.A[i, i] += self.alpha

    @ti.kernel
    def _solve_cg_kernel(self, b_vec: ti.template()):
        # Initialize
        self.x_vec.fill(0)
        for i in self.r_vec:
            self.r_vec[i] = b_vec[i]
            self.p_vec[i] = b_vec[i]

        rs_old = 0.0
        for i in self.r_vec:
            rs_old += self.r_vec[i] * self.r_vec[i]

        if ti.sqrt(rs_old) < 1e-9:
            return

        # Main CG loop
        for _ in range(self.n_iter):
            self.Ap_vec.fill(0)
            for i, j in self.A:
                ti.atomic_add(self.Ap_vec[i], self.A[i, j] * self.p_vec[j])

            pAp = 0.0
            for i in self.p_vec:
                pAp += self.p_vec[i] * self.Ap_vec[i]

            alpha_k = rs_old / pAp if pAp != 0 else 0.0

            for i in self.x_vec:
                self.x_vec[i] += alpha_k * self.p_vec[i]
                self.r_vec[i] -= alpha_k * self.Ap_vec[i]

            rs_new = 0.0
            for i in self.r_vec:
                rs_new += self.r_vec[i] * self.r_vec[i]

            if ti.sqrt(rs_new) < 1e-9:
                break

            beta = rs_new / rs_old
            for i in self.p_vec:
                self.p_vec[i] = self.r_vec[i] + beta * self.p_vec[i]
            rs_old = rs_new

    def fit(self, X_np, Y_np):
        n_reservoir, n_samples = X_np.shape
        n_output, _ = Y_np.shape
        print("  Initializing Taichi fields for CG solver...")
        X_ti = ti.field(dtype=ti.f32, shape=(n_reservoir, n_samples))
        B_np = (X_np @ Y_np.T).astype(np.float32)
        self.A = ti.field(dtype=ti.f32, shape=(n_reservoir, n_reservoir))
        b_vec, self.x_vec, self.r_vec, self.p_vec, self.Ap_vec = (
            ti.field(dtype=ti.f32, shape=n_reservoir) for _ in range(5)
        )
        X_ti.from_numpy(X_np)
        print("  Computing A = XX^T + alpha*I matrix...")
        self._compute_A(X_ti)
        W_out_np = np.zeros((n_output, n_reservoir), dtype=np.float32)
        print(f"  Solving for W_out using Conjugate Gradient ({self.n_iter} iterations)...")
        for j in range(n_output):
            b_vec.from_numpy(B_np[:, j])
            self._solve_cg_kernel(b_vec)
            W_out_np[j, :] = self.x_vec.to_numpy()
        return W_out_np


class NumpyRidge:
    """A Ridge Regression solver using the Conjugate Gradient method in NumPy."""

    def __init__(self, alpha=1e-4, n_iter=30):
        self.alpha = alpha
        self.n_iter = n_iter

    def _solve_cg(self, A, b):
        x = np.zeros_like(b, dtype=np.float32)
        r = b - (A @ x)
        p = r.copy()
        rs_old = np.dot(r, r)
        if np.sqrt(rs_old) < 1e-9:
            return x
        for _ in range(self.n_iter):
            Ap = A @ p
            pAp = np.dot(p, Ap)
            alpha_k = rs_old / pAp if pAp != 0 else 0.0
            x += alpha_k * p
            r -= alpha_k * Ap
            rs_new = np.dot(r, r)
            if np.sqrt(rs_new) < 1e-9:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def fit(self, X, Y):
        print("  Computing A = XX^T + alpha*I matrix...")
        A = X @ X.T
        A += self.alpha * np.identity(A.shape[0], dtype=np.float32)
        B = X @ Y.T
        n_output, n_reservoir = Y.shape[0], X.shape[0]
        W_out = np.zeros((n_output, n_reservoir), dtype=np.float32)
        print(f"  Solving for W_out using NumPy Conjugate Gradient ({self.n_iter} iterations)...")
        for j in range(n_output):
            b_vec = B[:, j]
            w_col = self._solve_cg(A, b_vec)
            W_out[j, :] = w_col
        return W_out
