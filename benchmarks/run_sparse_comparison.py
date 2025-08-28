import argparse
import csv
import os
import time

import numpy as np
import taichi as ti
from omegaconf import OmegaConf

# Import from the rcpy library
from rcpy.config import get_config
from rcpy.esn import EchoStateNetwork
from rcpy.sparse_esn import SparseEchoStateNetwork


def main():
    """
    Runs a single ESN benchmark configuration for either the dense or sparse
    implementation and appends the result to a CSV file.
    """
    # --- Configuration Loading ---
    parser = argparse.ArgumentParser(description="Run a single ESN benchmark.")
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file.")
    # Add a flag to select the ESN implementation
    parser.add_argument("--use_sparse", action="store_true", help="Use the SparseEchoStateNetwork.")
    args, unknown = parser.parse_known_args()

    conf = get_config()
    if args.config:
        file_conf = OmegaConf.load(args.config)
        conf = OmegaConf.merge(conf, file_conf)

    cli_conf = OmegaConf.from_cli(unknown)
    conf = OmegaConf.merge(conf, cli_conf)

    # --- Initialize Taichi ---
    if conf.taichi.backend.lower() == "gpu":
        ti.init(arch=ti.gpu)
    else:
        ti.init(arch=ti.cpu)
    print(f"Taichi backend initialized: {conf.taichi.backend}")

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(conf))
    print("---------------------")

    # --- Generate Data ---
    time_np = np.linspace(0, 80, conf.data.n_total_samples)
    clean_data = np.sin(time_np)
    noise = conf.data.noise_amplitude * np.random.randn(conf.data.n_total_samples)
    data = (clean_data + noise).reshape(-1, 1).astype(np.float32)

    input_data, target_data = data[:-1], data[1:]
    train_input = input_data[: conf.data.n_train_samples]
    train_target = target_data[: conf.data.n_train_samples]
    test_input = data[conf.data.n_train_samples : -1]
    test_target = data[conf.data.n_train_samples + 1 :]

    # --- Benchmarking ---
    start_init = time.perf_counter()
    if args.use_sparse:
        print("--- Using SparseEchoStateNetwork ---")
        esn = SparseEchoStateNetwork(conf)
        version_label = "sparse"
    else:
        print("--- Using EchoStateNetwork (Dense) ---")
        esn = EchoStateNetwork(conf)
        version_label = "dense"
    end_init = time.perf_counter()
    init_time = end_init - start_init

    start_fit = time.perf_counter()
    esn.fit(train_input, train_target, conf)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    start_predict = time.perf_counter()
    predictions = esn.predict(test_input, conf)
    end_predict = time.perf_counter()
    predict_time = end_predict - start_predict

    total_time = init_time + fit_time + predict_time
    mse = np.mean((predictions[: len(test_target)] - test_target) ** 2)

    # --- Save Results to CSV ---
    results_filename = conf.experiment.benchmark_output_file
    output_dir = os.path.dirname(results_filename)

    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    file_exists = os.path.exists(results_filename)

    data_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": version_label,
        "backend": conf.taichi.backend,
        "n_reservoir": conf.esn.n_reservoir,
        "sparsity": conf.esn.sparsity,
        "init_time": f"{init_time:.5f}",
        "fit_time": f"{fit_time:.5f}",
        "predict_time": f"{predict_time:.5f}",
        "total_time": f"{total_time:.5f}",
        "mse": f"{mse:.6f}",
    }

    try:
        with open(results_filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_row)
        print(f"\nBenchmark results appended to {results_filename}")
    except IOError as e:
        print(f"Error: Could not write to {results_filename}. Reason: {e}")


if __name__ == "__main__":
    main()
