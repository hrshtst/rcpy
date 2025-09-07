import argparse
import csv
import os
import time

import numpy as np
import taichi as ti
from omegaconf import OmegaConf

# Import from the rcpy library
from rcpy.config import get_config
from rcpy.esn import EchoStateNetwork, NumpyEchoStateNetwork


def main():
    """
    Runs a single ESN benchmark configuration and appends the result to a CSV file.
    """
    # --- Configuration Loading ---
    parser = argparse.ArgumentParser(description="Run a single ESN benchmark.")
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file.")
    args, unknown = parser.parse_known_args()

    conf = get_config()
    if args.config:
        file_conf = OmegaConf.load(args.config)
        conf = OmegaConf.merge(conf, file_conf)

    cli_conf = OmegaConf.from_cli(unknown)
    conf = OmegaConf.merge(conf, cli_conf)

    # --- Initialize Taichi if using the Taichi version ---
    if not conf.experiment.use_numpy_version:
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
    if conf.experiment.use_numpy_version:
        esn = NumpyEchoStateNetwork(conf)
    else:
        esn = EchoStateNetwork(conf)
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

    if conf.experiment.use_numpy_version:
        init_method = "numpy_pi" if conf.numpy_algos.use_power_iteration else "numpy_eigvals"
        ridge_solver_method = "numpy_cg" if conf.numpy_algos.use_conjugate_gradient else "numpy_solve"
        update_method = "numpy"
        predict_method = "numpy"
    else:
        init_method = "taichi_pi" if conf.esn.use_taichi_init else "numpy_eigvals"
        ridge_solver_method = "taichi_cg" if conf.solver.use_taichi_ridge else "numpy_pinv"
        update_method = "numpy_compute" if conf.experiment.use_numpy_update_in_taichi else "taichi_kernel"
        predict_method = "numpy_compute" if conf.experiment.use_numpy_predict_in_taichi else "taichi_kernel"

    data_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "numpy" if conf.experiment.use_numpy_version else "taichi",
        "backend": "cpu" if conf.experiment.use_numpy_version else conf.taichi.backend,
        "init_method": init_method,
        "ridge_solver": ridge_solver_method,
        "update_method": update_method,
        "predict_method": predict_method,
        "n_reservoir": conf.esn.n_reservoir,
        "sparsity": conf.esn.sparsity,
        "leaking_rate": conf.esn.leaking_rate,
        "spectral_radius": conf.esn.spectral_radius,
        "noise_amplitude": conf.data.noise_amplitude,
        "ridge_alpha": conf.solver.ridge_alpha,
        "cg_iterations": conf.solver.cg_iterations,
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
