#!/bin/bash
#
# This script runs a full benchmark suite for the ESN library,
# comparing multiple configurations across different reservoir sizes.
#

echo "--- Starting ESN Benchmark Suite ---"

# --- Configuration ---
NUM_RUNS=10
SIZES=(1000 2000 4000 8000)
MAIN_SCRIPT="benchmarks/run_and_save.py"
OUTPUT_DIR="benchmarks/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/benchmark_run_${TIMESTAMP}.csv"

# --- Pre-run checks ---
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: Main script '$MAIN_SCRIPT' not found. Make sure you are running this from the project root."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Results will be saved to: $OUTPUT_FILE"

# --- Main Benchmark Loop ---
for i in $(seq 1 $NUM_RUNS); do
  echo ""
  echo "--- Starting Run $i of $NUM_RUNS ---"
  for size in "${SIZES[@]}"; do
    echo "--- Testing Reservoir Size: $size (Run $i) ---"

    COMMON_ARGS="esn.n_reservoir=$size experiment.show_plot=false experiment.benchmark_output_file=$OUTPUT_FILE"

    # --- NumPy Configurations ---
    echo "  Running: NumPy (Standard Algos)"
    python "$MAIN_SCRIPT" experiment.use_numpy_version=true numpy_algos.use_power_iteration=false numpy_algos.use_conjugate_gradient=false $COMMON_ARGS

    echo "  Running: NumPy (PowerIter Init)"
    python "$MAIN_SCRIPT" experiment.use_numpy_version=true numpy_algos.use_power_iteration=true numpy_algos.use_conjugate_gradient=false $COMMON_ARGS

    # --- Taichi Configurations ---
    for backend in "gpu" "cpu"; do
      echo "  Running: Taichi-$backend (Taichi Predict)"
      python "$MAIN_SCRIPT" experiment.use_numpy_version=false taichi.backend=$backend experiment.use_numpy_predict_in_taichi=false $COMMON_ARGS

      echo "  Running: Taichi-$backend (NumPy Predict)"
      python "$MAIN_SCRIPT" experiment.use_numpy_version=false taichi.backend=$backend experiment.use_numpy_predict_in_taichi=true $COMMON_ARGS
    done
  done
done

echo ""
echo "--- Benchmark runs complete. ---"
echo "Results have been saved to $OUTPUT_FILE"
echo "You can now generate plots and tables by running:"
echo "python benchmarks/plot_results.py $OUTPUT_FILE"
