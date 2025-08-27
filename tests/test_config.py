import pytest
from omegaconf import DictConfig, OmegaConf  # Import DictConfig

from rcpy.config import ExperimentConfig, get_config


def test_get_config_returns_correct_type():
    """
    Tests if the get_config function returns a valid OmegaConf object
    and that its structure matches the ExperimentConfig dataclass.
    """
    conf = get_config()
    # FIX: Assert that the instance is of type DictConfig
    assert isinstance(conf, DictConfig)
    # Check if the structure is valid by trying to instantiate the dataclass from it
    try:
        ExperimentConfig(**conf)
    except TypeError as e:
        pytest.fail(f"Default configuration does not match ExperimentConfig structure: {e}")
    print("\nConfig type and structure test passed.")


def test_default_values():
    """
    Tests if some key default values are set as expected.
    """
    conf = get_config()
    assert conf.experiment.use_numpy_version is False
    assert conf.taichi.backend == "gpu"
    assert conf.esn.n_reservoir == 2000
    print("Default values test passed.")


def test_config_merging():
    """
    Tests if configurations are merged correctly, following the hierarchy:
    Default < YAML < Command-line
    """
    # 1. Start with the default config
    conf = get_config()

    # 2. Create a mock YAML configuration to merge
    yaml_str = """
    esn:
      n_reservoir: 5000
      sparsity: 0.99
    taichi:
      backend: cpu
    """
    yaml_conf = OmegaConf.create(yaml_str)

    # Merge the YAML config
    conf = OmegaConf.merge(conf, yaml_conf)

    # Assert that YAML values have overridden defaults
    assert conf.esn.n_reservoir == 5000
    assert conf.esn.sparsity == 0.99
    assert conf.taichi.backend == "cpu"
    # Assert that a non-overridden value remains the default
    assert conf.esn.leaking_rate == 0.2

    # 3. Create a mock command-line configuration to merge
    cli_str = "esn.n_reservoir=8000 experiment.show_plot=false"
    cli_conf = OmegaConf.from_dotlist(cli_str.split())

    # Merge the CLI config
    conf = OmegaConf.merge(conf, cli_conf)

    # Assert that CLI values have overridden YAML and default values
    assert conf.esn.n_reservoir == 8000
    assert conf.experiment.show_plot is False
    # Assert that the YAML value not overridden by CLI persists
    assert conf.esn.sparsity == 0.99
    print("Config merging hierarchy test passed.")
