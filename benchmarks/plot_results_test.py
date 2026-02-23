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
    Main function to load data and generate all plots and tables.
    """
    parser = argparse.ArgumentParser(description="Generate plots and LaTeX tables from ESN benchmark results.")
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
            init = "PowerIter" if row["init_method"] == "taichi_pi" else "EigVals"
            solver = "CG" if row["ridge_solver"] == "taichi_cg" else "Pinv"
            update = "TaichiUpdate" if row["update_method"] == "taichi_update" else "NumPyUpdate"
            predict = "TaichiKernel" if row["predict_method"] == "taichi_kernel" else "NumPyLoop"
            conditions.append(f"Taichi ({init}, {solver}, {update}, {predict})")
    df["Configuration"] = conditions

    print("Data loaded and processed. Generating outputs...")

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

    # Combined Performance Plot
    fig_perf, axes_perf = plt.subplots(2, 2, figsize=(20, 15))
    fig_perf.suptitle("ESN Benchmark Performance\n(Lines are Mean, Shaded areas are 95% CI)", y=0.97)
    for i, (metric, title) in enumerate(time_metrics):
        ax = axes_perf.flatten()[i]
        sns.lineplot(data=df, x="n_reservoir", y=metric, hue="Configuration", style="Configuration", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Reservoir Size")
        ax.set_ylabel("Time (seconds)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(title="Configuration")
        ax.grid(True, which="both", ls="--")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    perf_output_filename = os.path.join(output_dir, f"benchmark_performance_summary{suffix}")
    plt.savefig(perf_output_filename)
    print(f"\nCombined performance plot saved to '{perf_output_filename}'")

    # Individual Performance Plots
    for metric, title in time_metrics:
        fig_single, ax_single = plt.subplots(figsize=(12, 8))
        sns.lineplot(
            data=df, x="n_reservoir", y=metric, hue="Configuration", style="Configuration", marker="o", ax=ax_single
        )
        ax_single.set_title(f"ESN Benchmark: {title}\n(Lines are Mean, Shaded areas are 95% CI)")
        ax_single.set_xlabel("Reservoir Size (n_reservoir)")
        ax_single.set_ylabel("Time (seconds)")
        ax_single.set_xscale("log", base=2)
        ax_single.set_yscale("log")
        ax_single.legend(title="Configuration")
        ax_single.grid(True, which="both", ls="--")
        plt.tight_layout()
        single_filename = os.path.join(output_dir, f"benchmark_{metric}{suffix}")
        plt.savefig(single_filename)
        print(f"Individual plot saved to '{single_filename}'")
        plt.close(fig_single)

    # MSE Plot
    fig_mse, ax_mse = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=df, x="n_reservoir", y="mse", hue="Configuration", style="Configuration", marker="o", ax=ax_mse)
    ax_mse.set_title("Prediction Accuracy (MSE) vs. Reservoir Size")
    ax_mse.set_xlabel("Reservoir Size")
    ax_mse.set_ylabel("Mean Squared Error (MSE)")
    ax_mse.set_xscale("log", base=2)
    ax_mse.legend(title="Configuration")
    ax_mse.grid(True, which="both", ls="--")
    plt.tight_layout()
    mse_output_filename = os.path.join(output_dir, f"benchmark_mse_summary{suffix}")
    plt.savefig(mse_output_filename)
    print(f"MSE plot saved to '{mse_output_filename}'")

    # Focused MSE Plot
    fig_focused_mse, ax_focused_mse = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=df, x="n_reservoir", y="mse", hue="Configuration", style="Configuration", marker="o", ax=ax_focused_mse
    )
    ax_focused_mse.set_title("Prediction Accuracy (MSE) vs. Reservoir Size")
    ax_focused_mse.set_xlabel("Reservoir Size")
    ax_focused_mse.set_ylabel("Mean Squared Error (MSE)")
    ax_focused_mse.set_xscale("log", base=2)
    ax_focused_mse.set_ylim(0, 0.02)
    ax_focused_mse.legend(title="Configuration")
    ax_focused_mse.grid(True, which="both", ls="--")
    plt.tight_layout()
    mse_output_filename = os.path.join(output_dir, f"benchmark_focused_mse_summary{suffix}")
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
