# src/utils.py
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_results(file_name: str, data: dict, timestamp: str, output_dir: str = "results") -> str:
    """Save the current run's results as a CSV in `output_dir` and return the file path."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base = os.path.splitext(file_name)[0]
    final_file_name = f"{base}_{timestamp}.csv"
    file_path = os.path.join(output_dir, final_file_name)

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")
    return file_path


def plot_portfolio_weights(
    qaoa_solution: np.ndarray,
    classical_solution: np.ndarray,
    file_name: str,
    timestamp: str,
    asset_names: list,
    output_dir: str = "results",
) -> str:
    """Create a bar plot comparing classical vs QAOA portfolio weights."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    idx = np.arange(len(asset_names))
    width = 0.35

    if classical_solution.sum() > 0:
        classical_weights = classical_solution / classical_solution.sum()
    else:
        classical_weights = np.zeros_like(classical_solution)

    if qaoa_solution.sum() > 0:
        qaoa_weights = qaoa_solution / qaoa_solution.sum()
    else:
        qaoa_weights = np.zeros_like(qaoa_solution)

    plt.figure(figsize=(10, 5))
    plt.bar(idx - width / 2, classical_weights, width, label="Classical")
    plt.bar(idx + width / 2, qaoa_weights, width, label="QAOA")

    plt.xticks(idx, asset_names, rotation=45, ha="right")
    plt.ylabel("Weight")
    plt.title("Portfolio weights: classical vs QAOA")
    plt.legend()
    plt.tight_layout()

    base = os.path.splitext(file_name)[0]
    final_file_name = f"{base}_{timestamp}.png"
    file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(file_path)
    plt.close()
    print(f"Weights plot saved to {file_path}")
    return file_path


def _portfolio_summary(
    solution: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix,
) -> Tuple[float, float]:
    """Return (expected_return, risk) for a given 0/1 selection vector."""
    selection = np.asarray(solution, dtype=float)
    if selection.sum() == 0:
        return 0.0, 0.0

    weights = selection / selection.sum()

    if hasattr(covariance_matrix, "values"):
        cov = covariance_matrix.values
    else:
        cov = np.array(covariance_matrix)

    portfolio_return = float(weights @ expected_returns)
    portfolio_risk = float(np.sqrt(weights.T @ cov @ weights))
    return portfolio_return, portfolio_risk


def plot_efficient_frontier(
    expected_returns: np.ndarray,
    covariance_matrix,
    classical_solution: np.ndarray,
    qaoa_solution: np.ndarray,
    file_name: str,
    timestamp: str,
    output_dir: str = "results",
) -> str:
    """Plot random portfolios plus the classical and QAOA points."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_assets = len(expected_returns)
    if hasattr(covariance_matrix, "values"):
        cov = covariance_matrix.values
    else:
        cov = np.array(covariance_matrix)

    num_portfolios = 2000
    all_returns = []
    all_risks = []

    for _ in range(num_portfolios):
        weights = np.random.rand(num_assets)
        weights /= weights.sum()

        r = float(weights @ expected_returns)
        s = float(np.sqrt(weights.T @ cov @ weights))

        all_returns.append(r)
        all_risks.append(s)

    classical_return, classical_risk = _portfolio_summary(
        classical_solution, expected_returns, cov
    )
    qaoa_return, qaoa_risk = _portfolio_summary(
        qaoa_solution, expected_returns, cov
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(all_risks, all_returns, alpha=0.3, s=10, label="Random portfolios")
    if classical_solution.sum() > 0:
        plt.scatter(
            [classical_risk],
            [classical_return],
            marker="x",
            s=80,
            label="Classical",
        )
    if qaoa_solution.sum() > 0:
        plt.scatter(
            [qaoa_risk],
            [qaoa_return],
            marker="o",
            s=80,
            label="QAOA",
        )

    plt.xlabel("Risk (volatility)")
    plt.ylabel("Expected return")
    plt.title("Toy efficient frontier")
    plt.legend()
    plt.tight_layout()

    base = os.path.splitext(file_name)[0]
    final_file_name = f"{base}_{timestamp}.png"
    file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(file_path)
    plt.close()
    print(f"Efficient frontier plot saved to {file_path}")
    return file_path


def generate_comparison_report(
    report_file_path: str,
    current_results: pd.DataFrame,
    previous_results,
    timestamp: str,
    asset_names: list,
) -> None:
    """Generate a small Markdown report comparing this run to the previous one (if any)."""
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(f"# Portfolio optimisation run - {timestamp}\n\n")
        f.write("## 1. Solutions\n")
        f.write("### Portfolio selections\n")
        f.write(
            "Here is a comparison of the assets selected by the classical and QAOA solvers.\n\n"
        )

        f.write("| Asset | Classical Solution | QAOA Solution |\n")
        f.write("|-------|--------------------|---------------|\n")
        for idx, asset in enumerate(asset_names):
            classical_val = "✅" if current_results["Classical Solution"][idx] == 1 else "❌"
            qaoa_val = "✅" if current_results["QAOA Solution"][idx] == 1 else "❌"
            f.write(f"| {asset} | {classical_val} | {qaoa_val} |\n")
        f.write("\n")

        if previous_results is not None:
            f.write("## 2. Comparison with previous run\n")
            f.write(
                "This section shows how the selections changed compared to the previous run.\n\n"
            )

            current_classical_assets = ", ".join(
                asset_names[i]
                for i, x in enumerate(current_results["Classical Solution"])
                if x == 1
            )
            previous_classical_assets = ", ".join(
                asset_names[i]
                for i, x in enumerate(previous_results["Classical Solution"])
                if x == 1
            )

            current_qaoa_assets = ", ".join(
                asset_names[i]
                for i, x in enumerate(current_results["QAOA Solution"])
                if x == 1
            )
            previous_qaoa_assets = ", ".join(
                asset_names[i]
                for i, x in enumerate(previous_results["QAOA Solution"])
                if x == 1
            )

            f.write("### Classical solution change\n")
            f.write(f"- **Current run:** {current_classical_assets or 'none'}\n")
            f.write(f"- **Previous run:** {previous_classical_assets or 'none'}\n\n")

            f.write("### QAOA solution change\n")
            f.write(f"- **Current run:** {current_qaoa_assets or 'none'}\n")
            f.write(f"- **Previous run:** {previous_qaoa_assets or 'none'}\n\n")
        else:
            f.write("## 2. Comparison with previous run\n")
            f.write("No previous run found to compare against.\n\n")

    print(f"Comparison report generated at {report_file_path}")
