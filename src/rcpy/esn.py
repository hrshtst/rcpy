import numpy as np
import taichi as ti

from rcpy.ridge import NumpyRidge, TaichiRidge


@ti.data_oriented
class EchoStateNetwork:
    """Taichi-accelerated Echo State Network."""

    def __init__(self, conf):
        self.cfg = conf.esn

        self.W_in = ti.field(dtype=ti.f32, shape=(self.cfg.n_reservoir, self.cfg.n_input))
        self.W_res = ti.field(dtype=ti.f32, shape=(self.cfg.n_reservoir, self.cfg.n_reservoir))
        self.W_out = ti.field(dtype=ti.f32, shape=(self.cfg.n_output, self.cfg.n_reservoir))
        self.x = ti.field(dtype=ti.f32, shape=self.cfg.n_reservoir)

        # --- Select initialization method based on config ---
        if self.cfg.use_taichi_init:
            self._initialize_weights_taichi()
        else:
            self._initialize_weights_numpy()

    @ti.kernel
    def _generate_W_res_kernel(self):
        for i, j in self.W_res:
            if ti.random() > self.cfg.sparsity:
                self.W_res[i, j] = ti.random() * 2.0 - 1.0
            else:
                self.W_res[i, j] = 0.0

    @ti.kernel
    def _power_iteration_step(self, b_in: ti.template(), b_out: ti.template()):
        for i, j in self.W_res:
            ti.atomic_add(b_out[i], self.W_res[i, j] * b_in[j])

    @ti.kernel
    def _get_vec_norm(self, vec: ti.template()) -> ti.f32:
        norm_sq = 0.0
        for i in vec:
            norm_sq += vec[i] * vec[i]
        return ti.sqrt(norm_sq)

    @ti.kernel
    def _normalize_vec(self, vec: ti.template(), norm: ti.f32):
        for i in vec:
            vec[i] /= norm

    @ti.kernel
    def _scale_matrix(self, scale_factor: ti.f32):
        for i, j in self.W_res:
            self.W_res[i, j] *= scale_factor

    def _estimate_spectral_radius_taichi(self, n_iters=20):
        b_k = ti.field(dtype=ti.f32, shape=self.cfg.n_reservoir)
        b_k_next = ti.field(dtype=ti.f32, shape=self.cfg.n_reservoir)
        b_k.from_numpy(np.random.rand(self.cfg.n_reservoir).astype(np.float32))
        norm = self._get_vec_norm(b_k)
        if norm > 0:
            self._normalize_vec(b_k, norm)
        for _ in range(n_iters):
            b_k_next.fill(0)
            self._power_iteration_step(b_k, b_k_next)
            norm = self._get_vec_norm(b_k_next)
            if norm == 0:
                break
            self._normalize_vec(b_k_next, norm)
            b_k.copy_from(b_k_next)
        b_k_next.fill(0)
        self._power_iteration_step(b_k, b_k_next)
        return self._get_vec_norm(b_k_next)

    def _initialize_weights_taichi(self):
        print("Initializing ESN weights using Taichi...")
        w_in_np = (np.random.rand(self.cfg.n_reservoir, self.cfg.n_input) - 0.5).astype(np.float32)
        self.W_in.from_numpy(w_in_np)
        print("  Generating W_res on device...")
        self._generate_W_res_kernel()
        print("  Estimating spectral radius via Power Iteration...")
        current_spectral_radius = self._estimate_spectral_radius_taichi()
        print(f"  Estimated spectral radius: {current_spectral_radius:.4f}")
        if current_spectral_radius > 1e-9:
            self._scale_matrix(self.cfg.spectral_radius / current_spectral_radius)
        print("Initialization complete.")

    def _initialize_weights_numpy(self):
        """Initializes weights using NumPy (can be slow for large reservoirs)."""
        print("Initializing ESN weights using NumPy...")
        w_in_np = (np.random.rand(self.cfg.n_reservoir, self.cfg.n_input) - 0.5).astype(np.float32)
        self.W_in.from_numpy(w_in_np)

        w_res_np = (np.random.rand(self.cfg.n_reservoir, self.cfg.n_reservoir) - 0.5).astype(np.float32)
        w_res_np[np.random.rand(*w_res_np.shape) > self.cfg.sparsity] = 0.0

        print("  Calculating spectral radius with np.linalg.eigvals (can be slow)...")
        eigenvalues = np.linalg.eigvals(w_res_np)
        current_spectral_radius = np.max(np.abs(eigenvalues))

        if current_spectral_radius > 1e-9:
            w_res_np *= self.cfg.spectral_radius / current_spectral_radius

        self.W_res.from_numpy(w_res_np)
        print("Initialization complete.")

    @ti.kernel
    def _update_state_kernel(self, u_t: ti.template()):
        pre_activation = ti.Vector([0.0 for _ in range(self.cfg.n_reservoir)], dt=ti.f32)
        for i, j in self.W_res:
            pre_activation[i] += self.W_res[i, j] * self.x[j]
        for i in range(self.cfg.n_reservoir):
            for j in range(self.cfg.n_input):
                pre_activation[i] += self.W_in[i, j] * u_t[j]
        for i in range(self.cfg.n_reservoir):
            new_x_i = ti.tanh(pre_activation[i])
            self.x[i] = (1 - self.cfg.leaking_rate) * self.x[i] + self.cfg.leaking_rate * new_x_i

    @ti.kernel
    def _get_output_kernel(self) -> ti.types.vector(1, ti.f32):
        output = ti.Vector([0.0 for _ in range(self.cfg.n_output)], dt=ti.f32)
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

        print("  Collecting reservoir states...")
        if conf.experiment.use_numpy_update_in_taichi:
            W_res_np = self.W_res.to_numpy()
            W_in_np = self.W_in.to_numpy()
            x_np = self.x.to_numpy()
            for t in range(n_samples):
                u_t = train_input[t]
                pre_activation = W_res_np @ x_np + W_in_np @ u_t
                new_x = np.tanh(pre_activation)
                x_np = (1 - self.cfg.leaking_rate) * x_np + self.cfg.leaking_rate * new_x
                if t >= washout_period:
                    collected_states[t - washout_period] = x_np
            self.x.from_numpy(x_np)
        else:
            u_ti = ti.field(dtype=ti.f32, shape=self.cfg.n_input)
            self.x.fill(0)
            for t in range(n_samples):
                u_ti.from_numpy(train_input[t])
                self._update_state_kernel(u_ti)
                if t >= washout_period:
                    collected_states[t - washout_period] = self.x.to_numpy()
        print("  State collection complete.")

        X_T = collected_states.T
        Y_T = target_data[washout_period:].T

        if solver_cfg.use_taichi_ridge:
            print("\n--- Using Taichi Ridge Solver ---")
            ridge_solver = TaichiRidge(alpha=solver_cfg.ridge_alpha, n_iter=solver_cfg.cg_iterations)
            w_out_np = ridge_solver.fit(X_T, Y_T)
        else:
            print("\n--- Using NumPy linalg.pinv Solver ---")
            w_out_np = (Y_T @ np.linalg.pinv(X_T)).astype(np.float32)

        self.W_out.from_numpy(w_out_np)
        print("Training complete.")

    def predict(self, test_data, conf):
        n_samples = test_data.shape[0]
        predictions = np.zeros((n_samples, self.cfg.n_output), dtype=np.float32)

        if conf.experiment.use_numpy_predict_in_taichi:
            # --- NumPy compute path ---
            print(f"\nStarting prediction (Taichi class, NumPy compute)...")
            # Convert Taichi fields to NumPy arrays for computation
            W_res_np = self.W_res.to_numpy()
            W_in_np = self.W_in.to_numpy()
            W_out_np = self.W_out.to_numpy()
            x_np = self.x.to_numpy()

            for t in range(n_samples):
                u_t = test_data[t]
                pre_activation = W_res_np @ x_np + W_in_np @ u_t
                new_x = np.tanh(pre_activation)
                x_np = (1 - self.cfg.leaking_rate) * x_np + self.cfg.leaking_rate * new_x
                y_t = W_out_np @ x_np
                predictions[t] = y_t

            # Update the internal Taichi state to match the final NumPy state
            self.x.from_numpy(x_np)

        else:
            # --- Taichi compute path ---
            print(f"\nStarting prediction (Taichi Kernel-per-step)...")
            u_ti = ti.field(dtype=ti.f32, shape=self.cfg.n_input)
            for t in range(n_samples):
                u_ti.from_numpy(test_data[t])
                self._update_state_kernel(u_ti)
                predictions[t] = self._get_output_kernel().to_numpy()

        print("  Prediction complete.")
        return predictions


class NumpyEchoStateNetwork:
    """Pure NumPy Echo State Network."""

    def __init__(self, conf):
        self.cfg = conf.esn
        self.solver_cfg = conf.solver
        self.numpy_algos_cfg = conf.numpy_algos
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.x = np.zeros(self.cfg.n_reservoir, dtype=np.float32)
        self._initialize_weights()

    def _estimate_spectral_radius_power_iteration(self, W, n_iters=20):
        b_k = np.random.rand(W.shape[1]).astype(np.float32)
        for _ in range(n_iters):
            b_k_next = W @ b_k
            norm = np.linalg.norm(b_k_next)
            if norm == 0:
                return 0.0
            b_k = b_k_next / norm
        return np.linalg.norm(W @ b_k)

    def _initialize_weights(self):
        print("Initializing ESN weights using NumPy...")
        self.W_in = (np.random.rand(self.cfg.n_reservoir, self.cfg.n_input) - 0.5).astype(np.float32)
        W_res_np = (np.random.rand(self.cfg.n_reservoir, self.cfg.n_reservoir) - 0.5).astype(np.float32)
        W_res_np[np.random.rand(*W_res_np.shape) > self.cfg.sparsity] = 0.0

        if self.numpy_algos_cfg.use_power_iteration:
            print("  Estimating spectral radius via Power Iteration (NumPy)...")
            current_spectral_radius = self._estimate_spectral_radius_power_iteration(W_res_np)
        else:
            print("  Calculating spectral radius with np.linalg.eigvals...")
            eigenvalues = np.linalg.eigvals(W_res_np)
            current_spectral_radius = np.max(np.abs(eigenvalues))

        print(f"  Estimated spectral radius: {current_spectral_radius:.4f}")
        if current_spectral_radius > 1e-9:
            W_res_np *= self.cfg.spectral_radius / current_spectral_radius
        self.W_res = W_res_np
        print("Initialization complete.")

    def fit(self, train_input, target_data, conf):
        washout_period = conf.data.washout_period
        print(f"\nStarting training with washout period of {washout_period}...")
        n_samples = train_input.shape[0]

        print("  Collecting reservoir states...")
        collected_states = np.zeros((n_samples - washout_period, self.cfg.n_reservoir), dtype=np.float32)
        self.x.fill(0)
        for t in range(n_samples):
            u_t = train_input[t]
            pre_activation = self.W_res @ self.x + self.W_in @ u_t
            new_x = np.tanh(pre_activation)
            self.x = (1 - self.cfg.leaking_rate) * self.x + self.cfg.leaking_rate * new_x
            if t >= washout_period:
                collected_states[t - washout_period] = self.x
        print("  State collection complete.")

        X = collected_states
        Y = target_data[washout_period:]

        if self.numpy_algos_cfg.use_conjugate_gradient:
            print("\n--- Using NumPy Ridge Solver (Conjugate Gradient) ---")
            ridge_solver = NumpyRidge(alpha=self.solver_cfg.ridge_alpha, n_iter=self.solver_cfg.cg_iterations)
            self.W_out = ridge_solver.fit(X.T, Y.T)
        else:
            print("\n--- Solving for W_out using np.linalg.solve ---")
            A = X.T @ X
            A += self.solver_cfg.ridge_alpha * np.identity(A.shape[0], dtype=np.float32)
            B = X.T @ Y
            W_out_T = np.linalg.solve(A, B)
            self.W_out = W_out_T.T

        print("Training complete.")

    def predict(self, test_data, conf):
        print(f"\nStarting prediction (NumPy Version)...")
        n_samples = test_data.shape[0]
        predictions = np.zeros((n_samples, self.cfg.n_output), dtype=np.float32)
        for t in range(n_samples):
            u_t = test_data[t]
            pre_activation = self.W_res @ self.x + self.W_in @ u_t
            new_x = np.tanh(pre_activation)
            self.x = (1 - self.cfg.leaking_rate) * self.x + self.cfg.leaking_rate * new_x
            y_t = self.W_out @ self.x
            predictions[t] = y_t
        print("  Prediction complete.")
        return predictions
