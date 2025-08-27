#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
from omegaconf import OmegaConf

from rcpy.config import get_config
from rcpy.esn import EchoStateNetwork, NumpyEchoStateNetwork


def main():
    """
    An example script showing how to use the ESN library to train a model
    on a noisy sine wave prediction task.
    """
    # --- 1. Load and customize configuration ---
    conf = get_config()

    # Example of overriding a parameter
    conf.esn.n_reservoir = 1000  # Use a smaller reservoir for a quick example run
    conf.experiment.use_numpy_version = False  # Set to True to test the NumPy version

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(conf))
    print("---------------------")

    # --- 2. Initialize Taichi if using the Taichi version ---
    if not conf.experiment.use_numpy_version:
        ti.init(arch=conf.taichi.backend)
        print(f"Taichi backend initialized: {conf.taichi.backend}")

    # --- 3. Generate Data ---
    print("--- Generating Data ---")
    time_np = np.linspace(0, 80, conf.data.n_total_samples)
    clean_data = np.sin(time_np)
    noise = conf.data.noise_amplitude * np.random.randn(conf.data.n_total_samples)
    data = (clean_data + noise).reshape(-1, 1).astype(np.float32)

    input_data, target_data = data[:-1], data[1:]
    train_input = input_data[: conf.data.n_train_samples]
    train_target = target_data[: conf.data.n_train_samples]
    test_input = data[conf.data.n_train_samples : -1]
    test_target = data[conf.data.n_train_samples + 1 :]

    # --- 4. Instantiate, Train, and Predict ---
    if conf.experiment.use_numpy_version:
        esn = NumpyEchoStateNetwork(conf)
    else:
        esn = EchoStateNetwork(conf)

    esn.fit(train_input, train_target, conf)
    predictions = esn.predict(test_input, conf)

    # --- 5. Plot Results ---
    if conf.experiment.show_plot:
        print("\nPlotting results...")
        mse = np.mean((predictions[: len(test_target)] - test_target) ** 2)
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(15, 6))
        plot_range = range(min(200, len(test_target)))
        ax.plot(test_target[plot_range], "b", label="True Target (with noise)", linewidth=2, alpha=0.7)
        ax.plot(predictions[plot_range], "r--", label="ESN Prediction", linewidth=2)
        ax.set_title("Echo State Network: Noisy Sine Wave Prediction (Test Set)", fontsize=16)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.text(
            0.02,
            0.1,
            f"MSE: {mse:.6f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
