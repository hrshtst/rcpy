import numpy as np
import scipy.sparse
import taichi as ti

from rcpy.ridge import TaichiRidge


@ti.data_oriented
class SparseEchoStateNetwork:
    """
    Taichi-accelerated Echo State Network with a sparse reservoir weight matrix.
    This implementation is optimized for reservoirs with high sparsity.
    """

    def __init__(self, conf):
        self.cfg = conf.esn
        self.dtype = ti.f32

        # Initialize fields for weights and reservoir state
        self.W_in = ti.field(dtype=self.dtype, shape=(self.cfg.n_reservoir, self.cfg.n_input))
        self.W_out = ti.field(dtype=self.dtype, shape=(self.cfg.n_output, self.cfg.n_reservoir))
        self.x = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)

        # Fields for sparse reservoir matrix in COO format
        self.W_res_rows = None
        self.W_res_cols = None
        self.W_res_vals = None

        self._initialize_weights()

    @ti.kernel
    def _get_vec_norm(self, vec: ti.template()) -> ti.f32:
        """Computes the L2 norm of a vector."""
        norm_sq = 0.0
        for i in vec:
            norm_sq += vec[i] * vec[i]
        return ti.sqrt(norm_sq)

    @ti.kernel
    def _normalize_vec(self, vec: ti.template(), norm: ti.f32):
        """Normalizes a vector in-place."""
        for i in vec:
            vec[i] /= norm

    @ti.kernel
    def _materialize_spmv(self, y: ti.template(), x: ti.template()):
        """
        Performs sparse matrix-vector product (y = W_res @ x) using COO format.
        """
        y.fill(0)
        for i in range(self.W_res_rows.shape[0]):
            row, col, val = self.W_res_rows[i], self.W_res_cols[i], self.W_res_vals[i]
            y[row] += val * x[col]

    def _estimate_spectral_radius_sparse(self, n_iters=20):
        """
        Estimates the spectral radius of the sparse W_res matrix using the power iteration method.
        """
        b_k = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)
        b_k_next = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)
        b_k.from_numpy(np.random.rand(self.cfg.n_reservoir).astype(np.float32))

        norm = self._get_vec_norm(b_k)
        if norm > 1e-9:
            self._normalize_vec(b_k, norm)

        for _ in range(n_iters):
            self._materialize_spmv(b_k_next, b_k)
            norm = self._get_vec_norm(b_k_next)
            if norm < 1e-9:
                return 0.0
            self._normalize_vec(b_k_next, norm)
            b_k.copy_from(b_k_next)

        self._materialize_spmv(b_k_next, b_k)
        return self._get_vec_norm(b_k_next)

    @ti.kernel
    def _scale_sparse_matrix_vals(self, val: ti.f32):
        """Kernel to scale the non-zero values of the sparse W_res matrix."""
        for i in self.W_res_vals:
            self.W_res_vals[i] *= val

    def _initialize_weights(self):
        """
        Initializes all ESN weights, including the sparse reservoir matrix,
        and scales its spectral radius.
        """
        print("Initializing sparse ESN weights using Taichi...")
        self.W_in.from_numpy((np.random.rand(self.cfg.n_reservoir, self.cfg.n_input) * 2 - 1).astype(np.float32))

        print("  Generating sparse W_res using SciPy...")
        n_reservoir = self.cfg.n_reservoir
        w_res_scipy = scipy.sparse.random(
            n_reservoir,
            n_reservoir,
            density=(1 - self.cfg.sparsity),
            format="coo",
            dtype=np.float32,
            random_state=np.random.RandomState(),
        )
        w_res_scipy.data = (w_res_scipy.data * 2) - 1

        self.W_res_rows = ti.field(dtype=ti.i32, shape=w_res_scipy.nnz)
        self.W_res_cols = ti.field(dtype=ti.i32, shape=w_res_scipy.nnz)
        self.W_res_vals = ti.field(dtype=self.dtype, shape=w_res_scipy.nnz)
        self.W_res_rows.from_numpy(w_res_scipy.row.astype(np.int32))
        self.W_res_cols.from_numpy(w_res_scipy.col.astype(np.int32))
        self.W_res_vals.from_numpy(w_res_scipy.data)

        print("  Estimating spectral radius of sparse W_res via Power Iteration...")
        current_spectral_radius = self._estimate_spectral_radius_sparse()
        print(f"  Estimated spectral radius: {current_spectral_radius:.4f}")

        if current_spectral_radius > 1e-9:
            scale_factor = self.cfg.spectral_radius / current_spectral_radius
            self._scale_sparse_matrix_vals(scale_factor)
            print(f"  W_res scaled by a factor of {scale_factor:.4f}.")
        print("Initialization complete.")

    @ti.kernel
    def _fit_kernel(self, train_input: ti.types.ndarray(), collected_states: ti.types.ndarray(), washout_period: int):
        for t in range(train_input.shape[0]):
            pre_activation = ti.Vector([0.0 for _ in range(self.cfg.n_reservoir)], dt=self.dtype)

            # Sparse reservoir update
            for i in range(self.W_res_rows.shape[0]):
                row, col, val = self.W_res_rows[i], self.W_res_cols[i], self.W_res_vals[i]
                pre_activation[row] += val * self.x[col]

            # Input update
            for i in range(self.cfg.n_reservoir):
                in_val = 0.0
                for j in range(self.cfg.n_input):
                    in_val += self.W_in[i, j] * train_input[t, j]
                pre_activation[i] += in_val

            # Leaky integration
            for i in range(self.cfg.n_reservoir):
                new_x_i = ti.tanh(pre_activation[i])
                self.x[i] = (1 - self.cfg.leaking_rate) * self.x[i] + self.cfg.leaking_rate * new_x_i

            if t >= washout_period:
                for i in range(self.cfg.n_reservoir):
                    collected_states[t - washout_period, i] = self.x[i]

    def fit(self, train_input, target_data, conf):
        washout_period = conf.data.washout_period
        solver_cfg = conf.solver
        print(f"\nStarting training with washout period of {washout_period}...")

        n_samples = train_input.shape[0]
        collected_states = np.zeros((n_samples - washout_period, self.cfg.n_reservoir), dtype=np.float32)

        self.x.fill(0)
        self._fit_kernel(train_input, collected_states, washout_period)

        print("  State collection complete.")

        X_T = collected_states.T
        Y_T = target_data[washout_period:].T

        print("\n--- Using Taichi Ridge Solver ---")
        ridge_solver = TaichiRidge(alpha=solver_cfg.ridge_alpha, n_iter=solver_cfg.cg_iterations)
        w_out_np = ridge_solver.fit(X_T, Y_T)

        self.W_out.from_numpy(w_out_np)
        print("Training complete.")

    @ti.kernel
    def _predict_kernel(self, test_data: ti.types.ndarray(), predictions: ti.types.ndarray()):
        for t in range(test_data.shape[0]):
            pre_activation = ti.Vector([0.0 for _ in range(self.cfg.n_reservoir)], dt=self.dtype)

            # Sparse reservoir update
            for i in range(self.W_res_rows.shape[0]):
                row, col, val = self.W_res_rows[i], self.W_res_cols[i], self.W_res_vals[i]
                pre_activation[row] += val * self.x[col]

            # Input update
            for i in range(self.cfg.n_reservoir):
                in_val = 0.0
                for j in range(self.cfg.n_input):
                    in_val += self.W_in[i, j] * test_data[t, j]
                pre_activation[i] += in_val

            # Leaky integration
            for i in range(self.cfg.n_reservoir):
                new_x_i = ti.tanh(pre_activation[i])
                self.x[i] = (1 - self.cfg.leaking_rate) * self.x[i] + self.cfg.leaking_rate * new_x_i

            # Output
            output = ti.Vector([0.0 for _ in range(self.cfg.n_output)], dt=self.dtype)
            for i in range(self.cfg.n_output):
                for j in range(self.cfg.n_reservoir):
                    output[i] += self.W_out[i, j] * self.x[j]

            for i in range(self.cfg.n_output):
                predictions[t, i] = output[i]

    def predict(self, test_data, conf):
        n_samples = test_data.shape[0]
        predictions = np.zeros((n_samples, self.cfg.n_output), dtype=np.float32)

        print(f"\nStarting prediction (Sparse Taichi ESN)...")
        self._predict_kernel(test_data, predictions)

        print("  Prediction complete.")
        return predictions
