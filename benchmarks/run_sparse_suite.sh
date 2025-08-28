#!/bin/bash
#
# This script runs a benchmark suite comparing the performance of the
# dense EchoStateNetwork and the new SparseEchoStateNetwork.
#

echo "--- Starting Dense vs. Sparse ESN Benchmark Suite ---"

# --- Configuration ---
NUM_RUNS=5
SIZES=(1000 2000 4000 8000)
SPARSITIES=(0.9 0.95 0.98 0.99 0.995)
MAIN_SCRIPT="benchmarks/run_sparse_comparison.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmarks/results/sparse_comparison_${TIMESTAMP}"
OUTPUT_FILE="${OUTPUT_DIR}/sparse_comparison_results.csv"

# --- Pre-run checks ---
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: Main script '$MAIN_SCRIPT' not found. Make sure you are running this from the project root 'rcpy/'."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Results will be saved to: $OUTPUT_FILE"

# --- Main Benchmark Loop ---
for i in $(seq 1 $NUM_RUNS); do
  echo ""
  echo "--- Starting Run $i of $NUM_RUNS ---"
  for size in "${SIZES[@]}"; do
    for sparsity in "${SPARSITIES[@]}"; do
      echo "--- Testing Size: $size, Sparsity: $sparsity (Run $i) ---"

      COMMON_ARGS="esn.n_reservoir=$size esn.sparsity=$sparsity experiment.show_plot=false experiment.benchmark_output_file=$OUTPUT_FILE"

      # --- Run Dense ESN ---
      echo "  Running: Dense ESN"
      uv run python "$MAIN_SCRIPT" $COMMON_ARGS

      # --- Run Sparse ESN ---
      echo "  Running: Sparse ESN"
      uv run python "$MAIN_SCRIPT" --use_sparse $COMMON_ARGS

    done
  done
done

echo ""
echo "--- Benchmark runs complete. ---"
echo "Results have been saved to $OUTPUT_FILE"
echo "You can now generate plots by running:"
echo "uv run python benchmarks/plot_sparse_results.py $OUTPUT_FILE"
