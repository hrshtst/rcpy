#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
from omegaconf import OmegaConf

# Assumes the rcpy library is installed or in the Python path
from rcpy.config import get_config
from rcpy.sparse_esn import SparseEchoStateNetwork


def main():
    """
    An example script showing how to use the SparseEchoStateNetwork to train a model
    on a noisy sine wave prediction task, with configurable parameters.
    """
    # --- 1. Configuration Loading ---
    parser = argparse.ArgumentParser(description="Run a single Sparse ESN example.")
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file.")
    # Add a specific argument for the plot output file
    parser.add_argument(
        "--plot_output_file",
        type=str,
        default="sparse_sine_wave_prediction.png",
        help="Path to save the output plot PNG file.",
    )

    args, unknown = parser.parse_known_args()

    conf = get_config()
    if args.config:
        file_conf = OmegaConf.load(args.config)
        conf = OmegaConf.merge(conf, file_conf)

    # OmegaConf handles all other command-line overrides
    cli_conf = OmegaConf.from_cli(unknown)
    conf = OmegaConf.merge(conf, cli_conf)

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(conf))
    print("---------------------")

    # --- 2. Initialize Taichi ---
    if conf.taichi.backend.lower() == "gpu":
        ti.init(arch=ti.gpu)
    else:
        ti.init(arch=ti.cpu)
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
    esn = SparseEchoStateNetwork(conf)
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
        ax.plot(predictions[plot_range], "r--", label="Sparse ESN Prediction", linewidth=2)
        ax.set_title("Sparse Echo State Network: Noisy Sine Wave Prediction (Test Set)", fontsize=16)
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

        # Save the figure to the file specified by the command-line argument
        output_filename = args.plot_output_file
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(output_filename)
        print(f"Plot saved to '{output_filename}'")

        plt.show()


if __name__ == "__main__":
    main()
