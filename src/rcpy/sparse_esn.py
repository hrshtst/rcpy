import numpy as np
import taichi as ti
from taichi.linalg import SparseMatrixBuilder

from rcpy.sparse_ridge import SparseTaichiRidge


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

    def _estimate_spectral_radius_sparse(self, n_iters=20):
        """
        Estimates the spectral radius of the sparse W_res matrix using the power iteration method.
        """
        b_k = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)
        b_k.from_numpy(np.random.rand(self.cfg.n_reservoir).astype(np.float32))

        # Normalize the initial random vector
        norm = self._get_vec_norm(b_k)
        if norm > 0:
            self._normalize_vec(b_k, norm)

        # Power iteration loop
        for _ in range(n_iters):
            b_k_next = self.W_res @ b_k
            norm = self._get_vec_norm(b_k_next)
            if norm == 0:
                return 0.0
            self._normalize_vec(b_k_next, norm)
            b_k.copy_from(b_k_next)

        # Calculate the final eigenvalue estimate
        w_b = self.W_res @ b_k
        return self._get_vec_norm(w_b)

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

        # Build the sparse reservoir weight matrix
        print("  Generating sparse W_res on device...")
        n_reservoir = self.cfg.n_reservoir
        # Estimate the number of non-zero elements
        num_triplets = int(n_reservoir * n_reservoir * (1 - self.cfg.sparsity)) + n_reservoir
        builder = SparseMatrixBuilder(n_reservoir, n_reservoir, max_num_triplets=num_triplets, dtype=self.dtype)

        @ti.kernel
        def build_W_res(builder: ti.template()):
            for _ in range(num_triplets):
                i, j = ti.floor(ti.random() * n_reservoir), ti.floor(ti.random() * n_reservoir)
                if ti.random() > self.cfg.sparsity:
                    builder[i, j] += ti.random() * 2.0 - 1.0

        build_W_res(builder)
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
        # Sparse matrix-vector product for the reservoir update
        pre_activation = self.W_res @ self.x

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

        X = collected_states
        Y = target_data[washout_period:]

        print("\n--- Using Sparse Taichi Ridge Solver ---")
        ridge_solver = SparseTaichiRidge(alpha=solver_cfg.ridge_alpha, n_iter=solver_cfg.cg_iterations)
        w_out_np = ridge_solver.fit(X.T, Y.T)
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
