from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse._base import _spbase

WeightsType = np.ndarray | csr_matrix | csc_matrix | coo_matrix | _spbase
WeightsTypeVar = TypeVar("WeightsTypeVar", np.ndarray, csr_matrix, csc_matrix, coo_matrix, _spbase)
SparsityType = Literal["dense", "csr", "csc", "coo"]
