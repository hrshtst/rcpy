import numpy as np
import taichi as ti
from taichi.linalg import SparseCG, SparseMatrixBuilder


@ti.data_oriented
class SparseTaichiRidge:
    """
    A Ridge Regression solver using the Conjugate Gradient method in Taichi for sparse matrices.

    Note: This implementation is designed for cases where the matrix A in AX=B is sparse.
    In the context of an ESN, A is computed as X @ X.T + alpha * I, where X is the matrix of
    reservoir states. Even with a sparse reservoir weight matrix, the collected states matrix X
    is generally dense, which in turn makes A dense. This class is provided for experimental
    purposes and for scenarios where A is known to be sparse.
    """

    def __init__(self, alpha=1e-4, n_iter=30):
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, X_np, Y_np):
        n_reservoir, n_samples = X_np.shape
        n_output, _ = Y_np.shape

        # Build the sparse matrix A = X @ X.T + alpha * I
        A_builder = SparseMatrixBuilder(n_reservoir, n_reservoir, max_num_triplets=n_reservoir * n_reservoir)

        @ti.kernel
        def compute_A(X: ti.types.ndarray(), builder: ti.template()):
            # Compute X @ X.T
            for i, j in ti.ndrange(n_reservoir, n_reservoir):
                sum_val = 0.0
                for k in range(n_samples):
                    sum_val += X[i, k] * X[j, k]
                if abs(sum_val) > 1e-9:  # Add only non-zero elements to the builder
                    builder[i, j] += sum_val
            # Add the regularization term alpha * I
            for i in range(n_reservoir):
                builder[i, i] += self.alpha

        compute_A(X_np, A_builder)
        A = A_builder.build()

        B_np = (X_np @ Y_np.T).astype(np.float32)
        W_out_np = np.zeros((n_output, n_reservoir), dtype=np.float32)

        print(f"  Solving for W_out using Sparse Conjugate Gradient ({self.n_iter} iterations)...")
        b = ti.field(dtype=ti.f32, shape=n_reservoir)

        # Solve for each output dimension
        for j in range(n_output):
            b.from_numpy(B_np[:, j].astype(np.float32))

            solver = SparseCG(A=A, b=b, max_iter=self.n_iter)
            x, success = solver.solve()

            if not success:
                print(f"Warning: Sparse CG did not converge for output dimension {j}")
            W_out_np[j, :] = x.to_numpy()

        return W_out_np
