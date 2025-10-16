# src/rcpy/esn.py
import numpy as np
import taichi as ti

from rcpy.lms import NumpyLMS, TaichiLMS
from rcpy.ridge import NumpyRidge, TaichiRidge
from rcpy.rls import NumpyRLS, TaichiRLS


@ti.data_oriented
class EchoStateNetwork:
    """Taichi-accelerated Echo State Network."""

    def __init__(self, conf, solver=None):
        self.cfg = conf.esn
        self.solver_cfg = conf.solver
        self.solver = solver
        self.dtype = ti.f32

        self.W_in = ti.field(dtype=self.dtype, shape=(self.cfg.n_reservoir, self.cfg.n_input))
        self.W_res = ti.field(dtype=self.dtype, shape=(self.cfg.n_reservoir, self.cfg.n_reservoir))
        self.W_out = ti.field(dtype=self.dtype, shape=(self.cfg.n_output, self.cfg.n_reservoir))
        self.x = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)

        # --- Select initialization method based on config ---
        if self.cfg.use_taichi_init:
            self._initialize_weights_taichi()
        else:
            self._initialize_weights_numpy()

        if self.solver is None:
            if self.solver_cfg.use_taichi_ridge:
                print("\n--- Using Taichi Ridge Solver ---")
                self.solver = TaichiRidge(alpha=self.solver_cfg.ridge_alpha, n_iter=self.solver_cfg.cg_iterations)
            else:
                print("\n--- Using NumPy linalg.pinv Solver ---")

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
        b_k = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)
        b_k_next = ti.field(dtype=self.dtype, shape=self.cfg.n_reservoir)
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
    def _update_state_and_collect_kernel(self, u: ti.template(), collected_states: ti.template(), washout_period: int):
        for t in range(u.shape[0]):
            pre_activation = ti.Vector([0.0 for _ in range(self.cfg.n_reservoir)], dt=self.dtype)
            for i, j in self.W_res:
                pre_activation[i] += self.W_res[i, j] * self.x[j]
            for i in ti.static(range(self.cfg.n_reservoir)):
                for j in ti.static(range(self.cfg.n_input)):
                    pre_activation[i] += self.W_in[i, j] * u[t, j]
            for i in ti.static(range(self.cfg.n_reservoir)):
                new_x_i = ti.tanh(pre_activation[i])
                self.x[i] = (1 - self.cfg.leaking_rate) * self.x[i] + self.cfg.leaking_rate * new_x_i
            if t >= washout_period:
                for i in ti.static(range(self.cfg.n_reservoir)):
                    collected_states[t - washout_period, i] = self.x[i]

    @ti.kernel
    def _predict_kernel(self, test_data: ti.template(), predictions: ti.template()):
        for t in range(test_data.shape[0]):
            # Update state
            pre_activation = ti.Vector([0.0 for _ in range(self.cfg.n_reservoir)], dt=self.dtype)
            for i, j in self.W_res:
                pre_activation[i] += self.W_res[i, j] * self.x[j]
            for i in ti.static(range(self.cfg.n_reservoir)):
                for j in ti.static(range(self.cfg.n_input)):
                    pre_activation[i] += self.W_in[i, j] * test_data[t, j]
            for i in ti.static(range(self.cfg.n_reservoir)):
                new_x_i = ti.tanh(pre_activation[i])
                self.x[i] = (1 - self.cfg.leaking_rate) * self.x[i] + self.cfg.leaking_rate * new_x_i

            # Get output
            for i in ti.static(range(self.cfg.n_output)):
                out = 0.0
                for j in ti.static(range(self.cfg.n_reservoir)):
                    out += self.W_out[i, j] * self.x[j]
                predictions[t, i] = out

    def fit(self, train_input, target_data, conf):
        washout_period = conf.data.washout_period
        print(f"\nStarting training with washout period of {washout_period}...")
        n_samples = train_input.shape[0]

        print("  Collecting reservoir states...")
        collected_states_ti = ti.field(dtype=self.dtype, shape=(n_samples - washout_period, self.cfg.n_reservoir))
        train_input_ti = ti.field(dtype=self.dtype, shape=train_input.shape)
        train_input_ti.from_numpy(train_input.astype(np.float32))

        self.x.fill(0)
        self._update_state_and_collect_kernel(train_input_ti, collected_states_ti, washout_period)
        print("  State collection complete.")

        collected_states = collected_states_ti.to_numpy()
        X_T = collected_states.T
        Y_T = target_data[washout_period:].T

        if self.solver is None:  # Special case for pinv
            w_out_np = (Y_T @ np.linalg.pinv(X_T)).astype(np.float32)
        else:
            w_out_np = self.solver.fit(X_T, Y_T)

        self.W_out.from_numpy(w_out_np)
        print("Training complete.")

    def predict(self, test_data, conf):
        n_samples = test_data.shape[0]
        predictions_ti = ti.field(dtype=self.dtype, shape=(n_samples, self.cfg.n_output))
        test_data_ti = ti.field(dtype=self.dtype, shape=test_data.shape)
        test_data_ti.from_numpy(test_data.astype(np.float32))

        print(f"\nStarting prediction (Taichi ESN)...")
        self._predict_kernel(test_data_ti, predictions_ti)

        print("  Prediction complete.")
        return predictions_ti.to_numpy()

    def predict_online(self, test_input, test_target):
        """Performs online prediction and learning."""
        n_samples = test_input.shape[0]
        predictions = np.zeros((n_samples, self.cfg.n_output), dtype=np.float32)

        u_ti = ti.field(dtype=ti.f32, shape=self.cfg.n_input)

        # Mini-batching for RLS
        batch_size = self.solver_cfg.rls_batch_size if isinstance(self.solver, (NumpyRLS, TaichiRLS)) else 1
        state_batch = []
        target_batch = []

        for t in range(n_samples):
            u_ti.from_numpy(test_input[t])
            # This part is not optimized as it is for online learning
            # and the performance bottleneck is likely in the solver update.
            # a kernel per step is acceptable here.
            pre_activation = self.W_res @ self.x + self.W_in @ u_ti
            new_x = np.tanh(pre_activation)
            self.x = (1 - self.cfg.leaking_rate) * self.x + self.cfg.leaking_rate * new_x

            # Predict
            predictions[t] = (self.W_out @ self.x).to_numpy()

            state_batch.append(self.x.to_numpy())
            target_batch.append(test_target[t])

            if len(state_batch) >= batch_size or t == n_samples - 1:
                X_batch_np = np.array(state_batch).T
                Y_batch_np = np.array(target_batch).T
                self.solver.update_batch(X_batch_np, Y_batch_np)
                self.W_out.from_numpy(self.solver.W_out.to_numpy())
                state_batch, target_batch = [], []

        return predictions


class NumpyEchoStateNetwork:
    """Pure NumPy Echo State Network."""

    def __init__(self, conf, solver=None):
        self.cfg = conf.esn
        self.solver_cfg = conf.solver
        self.numpy_algos_cfg = conf.numpy_algos
        self.solver = solver
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.x = np.zeros(self.cfg.n_reservoir, dtype=np.float32)
        self._initialize_weights()

        if self.solver is None:
            if self.numpy_algos_cfg.use_conjugate_gradient:
                print("\n--- Using NumPy Ridge Solver (Conjugate Gradient) ---")
                self.solver = NumpyRidge(alpha=self.solver_cfg.ridge_alpha, n_iter=self.solver_cfg.cg_iterations)
            else:
                print("\n--- Solving for W_out using np.linalg.solve ---")

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

        if self.solver is None:  # Special case for linalg.solve
            A = X.T @ X
            A += self.solver_cfg.ridge_alpha * np.identity(A.shape[0], dtype=np.float32)
            B = X.T @ Y
            W_out_T = np.linalg.solve(A, B)
            self.W_out = W_out_T.T
        else:
            self.W_out = self.solver.fit(X.T, Y.T)

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

    def predict_online(self, test_input, test_target):
        """Performs online prediction and learning."""
        n_samples = test_input.shape[0]
        predictions = np.zeros((n_samples, self.cfg.n_output), dtype=np.float32)

        batch_size = self.solver_cfg.rls_batch_size if isinstance(self.solver, NumpyRLS) else 1
        state_batch = []
        target_batch = []

        for t in range(n_samples):
            u_t = test_input[t]
            pre_activation = self.W_res @ self.x + self.W_in @ u_t
            new_x = np.tanh(pre_activation)
            self.x = (1 - self.cfg.leaking_rate) * self.x + self.cfg.leaking_rate * new_x

            # Predict
            y_t = self.W_out @ self.x
            predictions[t] = y_t

            state_batch.append(self.x)
            target_batch.append(test_target[t])

            if len(state_batch) >= batch_size or t == n_samples - 1:
                X_batch = np.array(state_batch).T
                Y_batch = np.array(target_batch).T
                self.solver.update_batch(X_batch, Y_batch)
                self.W_out = self.solver.W_out
                state_batch, target_batch = [], []

        return predictions
