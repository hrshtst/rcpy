import numpy as np
import scipy.sparse
import taichi as ti
from taichi.linalg import SparseMatrixBuilder

from rcpy.ridge import TaichiRidge  # Use the original dense solver


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
        self.W_res = None  # Will be initialized as a sparse matrix
        self.W_out = ti.field(dtype=self.dtype, shape=(self.cfg.n_output, self.cfg.n_reservoir))
        self.x = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)

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
    def _materialize_spmv(self, x: ti.template(), y: ti.template()):
        """
        Materializes the sparse matrix-vector product (self.W_res @ x) into a concrete field y.
        This is done by explicitly iterating over the sparse matrix elements.
        """
        y.fill(0)
        for i, j in self.W_res:
            y[i] += self.W_res[i, j] * x[j]

    def _estimate_spectral_radius_sparse(self, n_iters=20):
        """
        Estimates the spectral radius of the sparse W_res matrix using the power iteration method.
        """
        b_k = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)
        b_k_next = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)  # A concrete field to store the result
        b_k.from_numpy(np.random.rand(self.cfg.n_reservoir).astype(np.float32))

        # Normalize the initial random vector
        norm = self._get_vec_norm(b_k)
        if norm > 1e-9:
            self._normalize_vec(b_k, norm)

        # Power iteration loop
        for _ in range(n_iters):
            # Explicitly materialize the expression into b_k_next
            self._materialize_spmv(b_k, b_k_next)

            norm = self._get_vec_norm(b_k_next)
            if norm < 1e-9:
                return 0.0

            # Normalize the concrete field in-place, then copy back
            self._normalize_vec(b_k_next, norm)
            b_k.copy_from(b_k_next)

        # Calculate the final eigenvalue estimate
        self._materialize_spmv(b_k, b_k_next)
        return self._get_vec_norm(b_k_next)

    @ti.kernel
    def _scale_sparse_matrix_kernel(self, val: ti.f32):
        """Kernel to scale the non-zero elements of the sparse W_res matrix."""
        for i, j in self.W_res:
            self.W_res[i, j] *= val

    def _initialize_weights(self):
        """
        Initializes all ESN weights, including the sparse reservoir matrix,
        and scales its spectral radius.
        """
        print("Initializing sparse ESN weights using Taichi...")

        # Initialize input weights
        w_in_np = (np.random.rand(self.cfg.n_reservoir, self.cfg.n_input) * 2 - 1).astype(np.float32)
        self.W_in.from_numpy(w_in_np)

        # Build the sparse reservoir weight matrix using SciPy first
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
        w_res_scipy.data = (w_res_scipy.data * 2) - 1  # Shift from [0, 1] to [-1, 1]

        # Build the Taichi sparse matrix from the SciPy COO data
        builder = SparseMatrixBuilder(n_reservoir, n_reservoir, max_num_triplets=w_res_scipy.nnz, dtype=self.dtype)

        @ti.kernel
        def fill_builder_from_coo(
            rows: ti.types.ndarray(),
            cols: ti.types.ndarray(),
            vals: ti.types.ndarray(),
            builder: ti.types.sparse_matrix_builder(),
        ):
            for i in range(rows.shape[0]):
                builder[rows[i], cols[i]] += vals[i]

        fill_builder_from_coo(w_res_scipy.row, w_res_scipy.col, w_res_scipy.data, builder)
        self.W_res = builder.build()

        # Estimate and scale the spectral radius
        print("  Estimating spectral radius of sparse W_res via Power Iteration...")
        current_spectral_radius = self._estimate_spectral_radius_sparse()
        print(f"  Estimated spectral radius: {current_spectral_radius:.4f}")

        if current_spectral_radius > 1e-9:
            scale_factor = self.cfg.spectral_radius / current_spectral_radius
            self._scale_sparse_matrix_kernel(scale_factor)
            print(f"  W_res scaled by a factor of {scale_factor:.4f}.")
        print("Initialization complete.")

    @ti.kernel
    def _update_state_kernel(self, u_t: ti.template()):
        # Use the explicit SPMV kernel
        pre_activation = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)
        self._materialize_spmv(self.x, pre_activation)

        # Add input contribution
        for i in range(self.cfg.n_reservoir):
            in_val = 0.0
            for j in range(self.cfg.n_input):
                in_val += self.W_in[i, j] * u_t[j]
            pre_activation[i] += in_val

        # Apply activation function and leaking rate
        for i in range(self.cfg.n_reservoir):
            new_x_i = ti.tanh(pre_activation[i])
            self.x[i] = (1 - self.cfg.leaking_rate) * self.x[i] + self.cfg.leaking_rate * new_x_i

    @ti.kernel
    def _get_output_kernel(self) -> ti.types.vector(1, ti.f32):
        output = ti.Vector([0.0 for _ in range(self.cfg.n_output)], dt=self.dtype)
        for i in range(self.cfg.n_output):
            for j in range(self.cfg.n_reservoir):
                output[i] += self.W_out[i, j] * self.x[j]
        return output

    def fit(self, train_input, target_data, conf):
        washout_period = conf.data.washout_period
        solver_cfg = conf.solver
        print(f"\nStarting training with washout period of {washout_period}...")

        n_samples = train_input.shape[0]
        collected_states = np.zeros((n_samples - washout_period, self.cfg.n_reservoir), dtype=np.float32)
        u_ti = ti.field(dtype=self.dtype, shape=self.cfg.n_input)

        print("  Collecting reservoir states...")
        self.x.fill(0)
        for t in range(n_samples):
            u_ti.from_numpy(train_input[t])
            self._update_state_kernel(u_ti)
            if t >= washout_period:
                collected_states[t - washout_period] = self.x.to_numpy()
        print("  State collection complete.")

        X_T = collected_states.T
        Y_T = target_data[washout_period:].T

        # Use the original TaichiRidge for the dense readout problem
        print("\n--- Using Taichi Ridge Solver ---")
        ridge_solver = TaichiRidge(alpha=solver_cfg.ridge_alpha, n_iter=solver_cfg.cg_iterations)
        w_out_np = ridge_solver.fit(X_T, Y_T)

        self.W_out.from_numpy(w_out_np)
        print("Training complete.")

    def predict(self, test_data, conf):
        n_samples = test_data.shape[0]
        predictions = np.zeros((n_samples, self.cfg.n_output), dtype=np.float32)

        print(f"\nStarting prediction (Sparse Taichi ESN)...")
        u_ti = ti.field(dtype=self.dtype, shape=self.cfg.n_input)
        for t in range(n_samples):
            u_ti.from_numpy(test_data[t])
            self._update_state_kernel(u_ti)
            predictions[t] = self._get_output_kernel().to_numpy()

        print("  Prediction complete.")
        return predictions
