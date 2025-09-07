#!/bin/bash
#
# This script runs a full benchmark suite for the rcpy library,
# comparing multiple configurations across different reservoir sizes.
#

echo "--- Starting ESN Benchmark Suite ---"

# --- Configuration ---
# NUM_RUNS=10
# SIZES=(100 200 400 800 1000 2000 4000 8000 10000 20000 40000)
NUM_RUNS=3
SIZES=(100 200 400 800 1000 2000 4000)
# Use the main script from the benchmarks directory
MAIN_SCRIPT="benchmarks/run.py"
# Generate a unique, timestamped directory for this benchmark run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmarks/results/benchmark_run_test_${TIMESTAMP}"
# Use a consistent filename within the timestamped directory
OUTPUT_FILE="${OUTPUT_DIR}/benchmark_results.csv"

# --- Pre-run checks ---
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: Main script '$MAIN_SCRIPT' not found. Make sure you are running this from the project root 'rcpy/'."
    exit 1
fi

# Create the output directory
mkdir -p "$OUTPUT_DIR"
echo "Results will be saved to: $OUTPUT_FILE"

# --- Main Benchmark Loop ---
for i in $(seq 1 $NUM_RUNS); do
  echo ""
  echo "--- Starting Run $i of $NUM_RUNS ---"
  for size in "${SIZES[@]}"; do
    echo "--- Testing Reservoir Size: $size (Run $i) ---"

    # Common arguments for all runs
    COMMON_ARGS="esn.n_reservoir=$size experiment.show_plot=false experiment.benchmark_output_file=$OUTPUT_FILE"

    # --- NumPy Configurations ---
    echo "  Running: NumPy (Standard Algos)"
    uv run python "$MAIN_SCRIPT" experiment.use_numpy_version=true numpy_algos.use_power_iteration=false numpy_algos.use_conjugate_gradient=false $COMMON_ARGS

    # # --- Taichi Configurations ---
    # echo "  Running: Taichi (Only reservoir update)"
    # uv run python "$MAIN_SCRIPT" experiment.use_numpy_version=false taichi.backend=gpu esn.use_taichi_init=false solver.use_taichi_ridge=false experiment.use_numpy_predict_in_taichi=false $COMMON_ARGS

    # echo "  Running: Taichi (Power iteration)"
    # uv run python "$MAIN_SCRIPT" experiment.use_numpy_version=false taichi.backend=gpu esn.use_taichi_init=true solver.use_taichi_ridge=false experiment.use_numpy_predict_in_taichi=false $COMMON_ARGS

    # echo "  Running: Taichi (CG method)"
    # uv run python "$MAIN_SCRIPT" experiment.use_numpy_version=false taichi.backend=gpu esn.use_taichi_init=true solver.use_taichi_ridge=true experiment.use_numpy_predict_in_taichi=false $COMMON_ARGS

    echo "  Running: Taichi (NumPy Predict)"
    uv run python "$MAIN_SCRIPT" experiment.use_numpy_version=false taichi.backend=gpu esn.use_taichi_init=true solver.use_taichi_ridge=true experiment.use_numpy_update_in_taichi=false experiment.use_numpy_predict_in_taichi=true $COMMON_ARGS

    echo "  Running: Taichi (NumPy Update)"
    uv run python "$MAIN_SCRIPT" experiment.use_numpy_version=false taichi.backend=gpu esn.use_taichi_init=true solver.use_taichi_ridge=true experiment.use_numpy_update_in_taichi=true  experiment.use_numpy_predict_in_taichi=true $COMMON_ARGS
  done
done

echo ""
echo "--- Benchmark runs complete. ---"
echo "Results have been saved to $OUTPUT_FILE"
echo "You can now generate plots and tables by running:"
echo "uv run python benchmarks/plot_results.py $OUTPUT_FILE"
