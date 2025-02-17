from __future__ import annotations

from enum import Enum, auto
from typing import Any, Final, Literal, TypeAlias, TypeVar

import numpy as np
from numpy.random import Generator
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse._base import _spbase

WeightsType: TypeAlias = np.ndarray | csr_matrix | csc_matrix | coo_matrix | _spbase
WeightsTypeVar = TypeVar("WeightsTypeVar", np.ndarray, csr_matrix, csc_matrix, coo_matrix, _spbase)
SparsityType: TypeAlias = Literal["dense", "csr", "csc", "coo"]

Unknown: TypeAlias = Any


class _NoDefault(Enum):
    """Enum to represent the absence of a default value in method parameters."""

    no_default = auto()


no_default: Final = _NoDefault.no_default
NoDefault: TypeAlias = Literal[_NoDefault.no_default]

SeedType: TypeAlias = int | Generator | NoDefault | None

# Local Variables:
# jinx-local-words: "Enum csc csr"
# End:
