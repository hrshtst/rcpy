import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    """
    Main function to load data and generate plots comparing dense and sparse ESNs.
    """
    parser = argparse.ArgumentParser(description="Generate plots from sparse ESN benchmark results.")
    parser.add_argument("csv_path", type=str, help="Path to the benchmark CSV file.")
    args = parser.parse_args()

    # --- 1. Load and Preprocess Data ---
    if not os.path.exists(args.csv_path):
        print(f"Error: Benchmark file not found at '{args.csv_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from '{args.csv_path}'...")
    df = pd.read_csv(args.csv_path)
    output_dir = os.path.dirname(args.csv_path) or "."

    # --- 2. Generate Plots ---
    generate_plots(df, output_dir)

    # --- 3. Generate LaTeX Table ---
    generate_latex_summary(df, output_dir)


def generate_plots(df, output_dir):
    """Generates and saves all performance plots."""
    sns.set_theme(style="whitegrid")
    time_metrics = [
        ("init_time", "Initialization Time"),
        ("fit_time", "Training Time"),
        ("predict_time", "Prediction Time"),
        ("total_time", "Total Time"),
    ]

    # Group by sparsity to create a separate plot for each level
    for sparsity, group in df.groupby("sparsity"):
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(
            f"Dense vs. Sparse ESN Performance (Sparsity = {sparsity})\n(Lines are Mean, Shaded areas are 95% CI)",
            fontsize=20,
            y=0.97,
        )

        for i, (metric, title) in enumerate(time_metrics):
            ax = axes.flatten()[i]
            sns.lineplot(data=group, x="n_reservoir", y=metric, hue="version", style="version", marker="o", ax=ax)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Reservoir Size", fontsize=12)
            ax.set_ylabel("Time (seconds)", fontsize=12)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.legend(title="Implementation", fontsize=10)
            ax.grid(True, which="both", ls="--")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        perf_output_filename = os.path.join(output_dir, f"performance_sparsity_{sparsity}.png")
        plt.savefig(perf_output_filename)
        print(f"\nPerformance plot for sparsity {sparsity} saved to '{perf_output_filename}'")
        plt.close(fig)

    print("\nAll plots generated.")
    plt.show()


def generate_latex_summary(df, output_dir):
    """Generates a complete LaTeX document with a summary table."""
    all_tables_latex = []
    metrics = ["init_time", "fit_time", "predict_time", "total_time", "mse"]

    # Group by sparsity, then by implementation version
    for sparsity, group in df.groupby("sparsity"):
        print(f"Generating LaTeX table for sparsity: {sparsity}")

        # Calculate mean and std for each version and reservoir size
        summary = group.groupby(["version", "n_reservoir"])[metrics].agg(["mean", "std"])

        # Format the table for readability
        formatted_df = pd.DataFrame()
        for version in group["version"].unique():
            for size in sorted(group["n_reservoir"].unique()):
                row_label = f"{version.capitalize()} (Size={size})"
                row_data = {}
                for metric in metrics:
                    mean_val = summary.loc[(version, size)][(metric, "mean")]
                    std_val = summary.loc[(version, size)][(metric, "std")]
                    row_data[metric] = f"{mean_val:.4f} $\\pm$ {std_val:.4f}"
                formatted_df[row_label] = pd.Series(row_data)

        formatted_df = formatted_df.T  # Transpose to have versions/sizes as rows
        formatted_df.columns = [col.replace("_", " ").title() for col in formatted_df.columns]

        latex_table_core = formatted_df.to_latex(
            caption=f"Performance Comparison for Sparsity = {sparsity} (mean $\\pm$ std). Times are in seconds.",
            label=f"tab:sparsity_{str(sparsity).replace('.', '')}",
            column_format="l" + "r" * len(metrics),
            position="H",
            escape=False,
        )
        all_tables_latex.append(latex_table_core)

    final_tables_string = "\n\\vspace{1cm}\n\n".join(all_tables_latex)

    latex_preamble = r"""
\documentclass[11pt]{article}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{float}
\usepackage{amsmath}   % For \pm command
\usepackage{pdflscape} % For landscape pages

\begin{document}
\begin{landscape}
\section*{ESN Performance Benchmark Results}
"""

    latex_postamble = r"""
\end{landscape}
\end{document}
"""

    full_latex_document = latex_preamble + final_tables_string + latex_postamble

    output_path = os.path.join(output_dir, "sparse_comparison_summary.tex")
    try:
        with open(output_path, "w") as f:
            f.write("% LaTeX document generated by plot_sparse_results.py\n")
            f.write("% To compile, run: pdflatex <filename>\n\n")
            f.write(full_latex_document)
        print(f"\nComplete LaTeX document successfully saved to '{output_path}'")
    except IOError as e:
        print(f"Error: Could not write to {output_path}. Reason: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
