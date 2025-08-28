import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    """
    Main function to load data and generate all plots and tables.
    """
    parser = argparse.ArgumentParser(description="Generate plots and LaTeX tables from ESN benchmark results.")
    parser.add_argument("csv_path", type=str, help="Path to the benchmark CSV file.")
    args = parser.parse_args()

    # --- 1. Load and Preprocess Data ---
    if not os.path.exists(args.csv_path):
        print(f"Error: Benchmark file not found at '{args.csv_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from '{args.csv_path}'...")
    df = pd.read_csv(args.csv_path)
    output_dir = os.path.dirname(args.csv_path) or "."

    # Create a descriptive 'Configuration' column for plotting legends
    conditions = []
    for _, row in df.iterrows():
        if row["version"] == "numpy":
            if row["init_method"] == "numpy_eigvals":
                conditions.append("NumPy (Standard Algos)")
            elif row["ridge_solver"] == "numpy_solve":
                conditions.append("NumPy (PowerIter Init)")
            else:
                conditions.append("NumPy (Fair Compare Algos)")
        else:
            backend = str(row["backend"]).upper()
            predict = "TaichiKernel" if row["predict_method"] == "taichi_kernel" else "NumPyLoop"
            conditions.append(f"Taichi-{backend} ({predict})")
    df["Configuration"] = conditions

    print("Data loaded and processed. Generating outputs...")

    # --- 2. Generate Plots ---
    generate_plots(df, output_dir)

    # --- 3. Generate LaTeX Table ---
    generate_latex_summary(df, output_dir)


def generate_plots(df, output_dir):
    """Generates and saves all performance and MSE plots."""
    sns.set_theme(style="whitegrid")
    time_metrics = [
        ("init_time", "Initialization Time"),
        ("fit_time", "Training Time"),
        ("predict_time", "Prediction Time"),
        ("total_time", "Total Time"),
    ]

    # Combined Performance Plot
    fig_perf, axes_perf = plt.subplots(2, 2, figsize=(20, 15))
    fig_perf.suptitle("ESN Benchmark Performance\n(Lines are Mean, Shaded areas are 95% CI)", fontsize=20, y=0.97)
    for i, (metric, title) in enumerate(time_metrics):
        ax = axes_perf.flatten()[i]
        sns.lineplot(data=df, x="n_reservoir", y=metric, hue="Configuration", style="Configuration", marker="o", ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Reservoir Size", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(title="Configuration", fontsize=10)
        ax.grid(True, which="both", ls="--")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    perf_output_filename = os.path.join(output_dir, "benchmark_performance_summary.png")
    plt.savefig(perf_output_filename)
    print(f"\nCombined performance plot saved to '{perf_output_filename}'")

    # Individual Performance Plots
    for metric, title in time_metrics:
        fig_single, ax_single = plt.subplots(figsize=(12, 8))
        sns.lineplot(
            data=df, x="n_reservoir", y=metric, hue="Configuration", style="Configuration", marker="o", ax=ax_single
        )
        ax_single.set_title(f"ESN Benchmark: {title}\n(Lines are Mean, Shaded areas are 95% CI)", fontsize=16)
        ax_single.set_xlabel("Reservoir Size", fontsize=12)
        ax_single.set_ylabel("Time (seconds)", fontsize=12)
        ax_single.set_xscale("log", base=2)
        ax_single.set_yscale("log")
        ax_single.legend(title="Configuration")
        ax_single.grid(True, which="both", ls="--")
        plt.tight_layout()
        single_filename = os.path.join(output_dir, f"benchmark_{metric}.png")
        plt.savefig(single_filename)
        print(f"Individual plot saved to '{single_filename}'")
        plt.close(fig_single)

    # MSE Plot
    fig_mse, ax_mse = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=df, x="n_reservoir", y="mse", hue="Configuration", style="Configuration", marker="o", ax=ax_mse)
    ax_mse.set_title("Prediction Accuracy (MSE) vs. Reservoir Size", fontsize=16)
    ax_mse.set_xlabel("Reservoir Size", fontsize=12)
    ax_mse.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
    ax_mse.set_xscale("log", base=2)
    ax_mse.legend(title="Configuration")
    ax_mse.grid(True, which="both", ls="--")
    plt.tight_layout()
    mse_output_filename = os.path.join(output_dir, "benchmark_mse_summary.png")
    plt.savefig(mse_output_filename)
    print(f"MSE plot saved to '{mse_output_filename}'")

    plt.show()


def generate_latex_summary(df, output_dir):
    """Generates a complete LaTeX document with a table for each configuration."""
    all_tables_latex = []
    unique_configs = sorted(df["Configuration"].unique())
    metrics = ["init_time", "fit_time", "predict_time", "total_time", "mse"]

    for config_name in unique_configs:
        print(f"Generating LaTeX table for: {config_name}")
        df_single_config = df[df["Configuration"] == config_name]
        grouped = df_single_config.groupby("n_reservoir")[metrics].agg(["mean", "std"])
        formatted_df = pd.DataFrame(index=grouped.index)
        for metric in metrics:
            mean_series = grouped[(metric, "mean")]
            std_series = grouped[(metric, "std")]
            formatted_df[metric] = mean_series.map("{:.5f}".format) + " $\\pm$ " + std_series.map("{:.5f}".format)

        formatted_df.columns = [col.replace("_", " ").title() for col in formatted_df.columns]
        formatted_df.index.name = "Reservoir Size"
        desired_order = ["Init Time", "Fit Time", "Predict Time", "Total Time", "Mse"]
        formatted_df = formatted_df[desired_order]

        latex_table_core = formatted_df.to_latex(
            caption=f"Performance for {config_name} (mean $\\pm$ std). Times are in seconds.",
            label=f"tab:{config_name.replace(' ', '').replace('(', '').replace(')', '').replace(',', '')}",
            column_format="l" + "r" * len(desired_order),
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
\usepackage{amsmath}

\begin{document}
\section*{ESN Performance Benchmark Results}
"""
    latex_postamble = r"""
\end{document}
"""
    full_latex_document = latex_preamble + final_tables_string + latex_postamble

    output_path = os.path.join(output_dir, "benchmark_summary_table.tex")
    try:
        with open(output_path, "w") as f:
            f.write("% LaTeX document generated by plot_results.py\n")
            f.write("% To compile, run: pdflatex <filename>\n\n")
            f.write(full_latex_document)
        print(f"\nComplete LaTeX document successfully saved to '{output_path}'")
    except IOError as e:
        print(f"Error: Could not write to {output_path}. Reason: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
