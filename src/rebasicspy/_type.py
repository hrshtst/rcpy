import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

WeightsType = np.ndarray | csr_matrix | csc_matrix | coo_matrix
