# src/rcpy/config.py
from dataclasses import dataclass, field

from omegaconf import OmegaConf


@dataclass
class ExperimentSetup:
    """High-level experiment settings."""

    use_numpy_version: bool = False
    use_numpy_update_in_taichi: bool = False
    use_numpy_predict_in_taichi: bool = False
    show_plot: bool = True
    benchmark_output_file: str = "benchmark_results.csv"


@dataclass
class TaichiConfig:
    """Configuration for the Taichi backend."""

    backend: str = "gpu"


@dataclass
class ESNConfig:
    """Configuration for the Echo State Network."""

    use_taichi_init: bool = True
    n_input: int = 1
    n_reservoir: int = 2000
    n_output: int = 1
    spectral_radius: float = 0.99
    leaking_rate: float = 0.2
    sparsity: float = 0.98


@dataclass
class DataConfig:
    """Configuration for the dataset."""

    n_total_samples: int = 2000
    n_train_samples: int = 1000
    noise_amplitude: float = 0.05
    washout_period: int = 100


@dataclass
class SolverConfig:
    """Configuration for the training solver."""

    solver_type: str = "ridge"  # "ridge" or "rls"
    # Ridge options
    use_taichi_ridge: bool = True
    ridge_alpha: float = 1e-4
    cg_iterations: int = 30
    # RLS options
    forgetting_factor: float = 0.98
    delta: float = 0.001


@dataclass
class NumpyAlgorithmicConfig:
    """Configuration for NumPy specific algorithms for fair comparison."""

    use_power_iteration: bool = False
    use_conjugate_gradient: bool = False


@dataclass
class ExperimentConfig:
    """Main configuration container."""

    experiment: ExperimentSetup = field(default_factory=ExperimentSetup)
    taichi: TaichiConfig = field(default_factory=TaichiConfig)
    esn: ESNConfig = field(default_factory=ESNConfig)
    data: DataConfig = field(default_factory=DataConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    numpy_algos: NumpyAlgorithmicConfig = field(default_factory=NumpyAlgorithmicConfig)


def get_config():
    """Creates a default configuration object."""
    return OmegaConf.structured(ExperimentConfig)
