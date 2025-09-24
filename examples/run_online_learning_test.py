# examples/run_online_learning_test.py
import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
from omegaconf import OmegaConf

from rcpy.config import get_config
from rcpy.esn import NumpyEchoStateNetwork
from rcpy.lms import NumpyLMS
from rcpy.rls import NumpyRLS


def main():
    """
    An example script demonstrating online learning with LMS and RLS
    to adapt to a changing sine wave frequency.
    """
    # --- 1. Configuration ---
    parser = argparse.ArgumentParser(description="Run ESN online learning example.")
    args, unknown = parser.parse_known_args()
    conf = get_config()
    cli_conf = OmegaConf.from_cli(unknown)
    conf = OmegaConf.merge(conf, cli_conf)

    # --- 2. Generate Data with Changing Dynamics ---
    print("--- Generating Data ---")
    n_total = 4000
    n_train = 2000
    change_point = 3000

    time_vec = np.linspace(0, 80, n_total)
    freq1, freq2 = 1.0, 1.5

    signal = np.zeros(n_total)
    signal[:change_point] = np.sin(freq1 * time_vec[:change_point])
    signal[change_point:] = np.sin(freq2 * time_vec[change_point:])

    data = signal.reshape(-1, 1).astype(np.float32)

    train_input = data[: n_train - 1]
    train_target = data[1:n_train]
    test_input = data[n_train - 1 : -1]
    test_target = data[n_train:]

    # --- 3. Offline Training (Ridge) ---
    print("--- 1. Offline Training with Ridge ---")
    conf.experiment.use_numpy_version = True
    conf.solver.solver_type = "ridge"
    esn_ridge = NumpyEchoStateNetwork(conf)
    esn_ridge.fit(train_input, train_target, conf)

    print("\n--- 2. Predicting with offline-trained model (no adaptation) ---")
    preds_ridge = esn_ridge.predict(test_input, conf)

    # --- 4. Online Adaptation (LMS) ---
    print("\n--- 3. Adapting online with LMS ---")
    esn_lms = copy.deepcopy(esn_ridge)
    esn_lms.solver_cfg.solver_type = "lms"
    esn_lms.solver = NumpyLMS(esn_lms.cfg.n_reservoir, esn_lms.cfg.n_output, learning_rate=0.01)
    esn_lms.W_out = esn_ridge.W_out.copy()  # Start with the same W_out
    preds_lms = esn_lms.predict_online(test_input, test_target)

    # --- 5. Online Adaptation (RLS) ---
    print("\n--- 4. Adapting online with RLS ---")
    esn_rls = copy.deepcopy(esn_ridge)
    esn_rls.solver_cfg.solver_type = "rls"
    esn_rls.solver = NumpyRLS(esn_rls.cfg.n_reservoir, esn_rls.cfg.n_output, forgetting_factor=0.99, delta=0.1)
    esn_rls.W_out = esn_ridge.W_out.copy()  # Start with the same W_out
    preds_rls = esn_rls.predict_online(test_input, test_target)

    # --- 6. Plot Results ---
    print("\n--- Plotting results ---")
    plt.figure(figsize=(15, 8))
    plt.plot(test_target, "k", label="True Signal", linewidth=2)
    plt.plot(preds_ridge, "b--", label="Ridge (No Adaptation)", alpha=0.8)
    plt.plot(preds_lms, "g-.", label="LMS (Online Adaptation)", alpha=0.8)
    plt.plot(preds_rls, "r:", label="RLS (Online Adaptation)", alpha=0.8)

    change_idx = change_point - n_train
    plt.axvline(x=change_idx, color="gray", linestyle="--", label="Frequency Change")

    plt.title("ESN Online Learning: Adapting to Changing Sine Wave Frequency", fontsize=16)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
