import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Plotting Configuration ---
SUPTITLE_SIZE = 26
TITLE_SIZE = 20
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 14


def main():
    """
    Main function to load data and generate plots comparing dense and sparse ESNs.
    """
    parser = argparse.ArgumentParser(description="Generate plots from sparse ESN benchmark results.")
    parser.add_argument("csv_path", type=str, help="Path to the benchmark CSV file.")
    parser.add_argument(
        "--suffix",
        type=str,
        default=".png",
        help="Suffix for the output plots (e.g., .png, .pdf, .svg). Default is .png.",
    )
    args = parser.parse_args()

    # Ensure suffix starts with a dot if not empty and doesn't already have one
    suffix = args.suffix
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix

    # --- 1. Load and Preprocess Data ---
    if not os.path.exists(args.csv_path):
        print(f"Error: Benchmark file not found at '{args.csv_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from '{args.csv_path}'...")
    df = pd.read_csv(args.csv_path)
    output_dir = os.path.dirname(args.csv_path) or "."

    # Add a connectivity column for more intuitive plotting
    df["connectivity"] = 1.0 - df["sparsity"]

    # --- 2. Generate Plots ---
    generate_plots(df, output_dir, suffix)

    # --- 3. Generate LaTeX Table ---
    generate_latex_summary(df, output_dir)


def generate_plots(df, output_dir, suffix):
    """Generates and saves all performance and MSE plots."""
    sns.set_theme(style="whitegrid")

    # Update global font sizes via rcParams
    plt.rcParams.update(
        {
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "legend.title_fontsize": LEGEND_SIZE,
            "figure.titlesize": SUPTITLE_SIZE,
        }
    )

    time_metrics = [
        ("init_time", "Initialization Time"),
        ("fit_time", "Training Time"),
        ("predict_time", "Prediction Time"),
        ("total_time", "Total Time"),
    ]

    # Group by connectivity to create a separate set of plots for each level
    for connectivity, group in df.groupby("connectivity"):
        # --- Combined Performance Plot ---
        fig_perf, axes_perf = plt.subplots(2, 2, figsize=(20, 15))
        fig_perf.suptitle(
            f"Dense vs. Sparse ESN Performance (Connectivity = {connectivity:.3f})\n(Lines are Mean, Shaded areas are 95% CI)",
            y=0.97,
        )
        for i, (metric, title) in enumerate(time_metrics):
            ax = axes_perf.flatten()[i]
            sns.lineplot(data=group, x="n_reservoir", y=metric, hue="version", style="version", marker="o", ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Reservoir Size")
            ax.set_ylabel("Time (seconds)")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.legend(title="Implementation")
            ax.grid(True, which="both", ls="--")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        perf_output_filename = os.path.join(
            output_dir, f"performance_summary_connectivity_{connectivity:.3f}{suffix}"
        )
        plt.savefig(perf_output_filename)
        print(f"\nCombined performance plot for connectivity {connectivity:.3f} saved to '{perf_output_filename}'")

        # --- Individual Performance Plots ---
        for metric, title in time_metrics:
            fig_single, ax_single = plt.subplots(figsize=(12, 8))
            sns.lineplot(
                data=group, x="n_reservoir", y=metric, hue="version", style="version", marker="o", ax=ax_single
            )
            ax_single.set_title(
                f"ESN Benchmark: {title} (Connectivity = {connectivity:.3f})\n(Lines are Mean, Shaded areas are 95% CI)"
            )
            ax_single.set_xlabel("Reservoir Size (n_reservoir)")
            ax_single.set_ylabel("Time (seconds)")
            ax_single.set_xscale("log", base=2)
            ax_single.set_yscale("log")
            ax_single.legend(title="Implementation")
            ax_single.grid(True, which="both", ls="--")
            plt.tight_layout()
            single_filename = os.path.join(
                output_dir, f"benchmark_{metric}_connectivity_{connectivity:.3f}{suffix}"
            )
            plt.savefig(single_filename)
            print(f"Individual plot saved to '{single_filename}'")
            plt.close(fig_single)

        # --- MSE Plot ---
        fig_mse, ax_mse = plt.subplots(figsize=(12, 8))
        sns.lineplot(data=group, x="n_reservoir", y="mse", hue="version", style="version", marker="o", ax=ax_mse)
        ax_mse.set_title(f"Prediction Accuracy (MSE) vs. Reservoir Size (Connectivity = {connectivity:.3f})")
        ax_mse.set_xlabel("Reservoir Size")
        ax_mse.set_ylabel("Mean Squared Error (MSE)")
        ax_mse.set_xscale("log", base=2)
        ax_mse.set_ylim(0.0, 0.005)
        ax_mse.legend(title="Implementation")
        ax_mse.grid(True, which="both", ls="--")
        plt.tight_layout()
        mse_output_filename = os.path.join(output_dir, f"benchmark_mse_connectivity_{connectivity:.3f}{suffix}")
        plt.savefig(mse_output_filename)
        print(f"MSE plot saved to '{mse_output_filename}'")

    print("\nAll plots generated.")
    plt.show()


def generate_latex_summary(df, output_dir):
    """Generates a complete LaTeX document with a summary table."""
    all_tables_latex = []
    metrics = ["init_time", "fit_time", "predict_time", "total_time", "mse"]

    # Group by connectivity, then by implementation version
    for connectivity, group in df.groupby("connectivity"):
        print(f"Generating LaTeX table for connectivity: {connectivity:.3f}")

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
            caption=f"Performance Comparison for Connectivity = {connectivity:.3f} (mean $\\pm$ std). Times are in seconds.",
            label=f"tab:connectivity_{str(connectivity).replace('.', 'p')}",
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
