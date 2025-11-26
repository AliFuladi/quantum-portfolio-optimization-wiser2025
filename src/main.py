# src/main.py
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.data_handler import fetch_stock_data, calculate_portfolio_metrics
from src.portfolio_model import create_quadratic_program, get_qubo
from src.solver import find_best_penalty, solve_with_classical_optimizer, solve_with_qaoa
from src.utils import (
    save_results,
    plot_portfolio_weights,
    plot_efficient_frontier,
    generate_comparison_report,
)


def main() -> None:
    """Run one end-to-end optimisation: data -> model -> classical + QAOA -> plots + report."""

    # --- Basic config for this toy project ---
    tickers = [
        "AAPL",
        "AMZN",
        "GOOG",
        "KO",
        "META",
        "MSFT",
        "NFLX",
        "NVDA",
        "SBUX",
        "TSLA",
    ]
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    # How many assets we want in the final portfolio
    num_assets_to_select = 4

    # Trade-off between return and risk in the objective
    risk_factor = 0.5

    print("=== Quantum / classical portfolio optimisation (toy model) ===")

    # --- Step 1: fetch market data ---
    print("\n1) Downloading historical data ...")
    price_data = fetch_stock_data(tickers, start_date, end_date)

    # --- Step 2: build simple return / risk estimates ---
    expected_returns, covariance_matrix, asset_names = calculate_portfolio_metrics(
        price_data, num_assets_to_select
    )

    # --- Step 3: formulate the quadratic program ---
    print("\n2) Building the optimisation model ...")
    qp = create_quadratic_program(
        expected_returns,
        covariance_matrix,
        num_assets_to_select,
        risk_factor,
    )

    # --- Step 4: solve classically ---
    print("\n3) Solving with a classical eigensolver ...")
    classical_solution = solve_with_classical_optimizer(qp)
    print("   Classical selection (0/1):", classical_solution.astype(int).tolist())

    # --- Step 5: try a QAOA-based solution ---
    print("\n4) Searching for a reasonable penalty for the QUBO encoding ...")
    penalty_values = np.linspace(5, 50, 10)
    best_penalty = find_best_penalty(qp, penalty_values)

    if best_penalty is not None:
        print(f"   Using penalty = {best_penalty}")
        qubo = get_qubo(qp, best_penalty)
        print("\n5) Running QAOA on the same problem ...")
        qaoa_solution = solve_with_qaoa(qubo)
        print("   QAOA selection (0/1):", qaoa_solution.astype(int).tolist())
    else:
        print("   Failed to find a sensible penalty; falling back to classical-only solution.")
        qaoa_solution = np.zeros_like(classical_solution)

    # --- Step 6: save results, plots and comparison report ---
    print("\n6) Saving results and generating plots ...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"

    results_data = {
        "Asset": asset_names,
        "Classical Solution": classical_solution.astype(int),
        "QAOA Solution": qaoa_solution.astype(int),
    }

    save_results("solutions", results_data, timestamp, output_dir=output_dir)
    plot_portfolio_weights(
        qaoa_solution,
        classical_solution,
        "weights_comparison",
        timestamp,
        asset_names,
        output_dir=output_dir,
    )
    plot_efficient_frontier(
        expected_returns,
        covariance_matrix,
        classical_solution,
        qaoa_solution,
        "efficient_frontier",
        timestamp,
        output_dir=output_dir,
    )

    # Build a tiny history-based report
    print("\n7) Generating Markdown comparison report ...")
    current_results_df = pd.DataFrame(results_data)
    previous_results_df = None

    # Look for previous 'solutions_*.csv' files
    solution_files = sorted(
        [
            f
            for f in os.listdir(output_dir)
            if f.startswith("solutions_") and f.endswith(".csv")
        ],
        reverse=True,
    )

    if len(solution_files) > 1:
        previous_file_path = os.path.join(output_dir, solution_files[1])
        try:
            previous_results_df = pd.read_csv(previous_file_path)
            print(f"Comparing against previous run in {previous_file_path}")
        except Exception as exc:
            print(f"Could not load previous results file for comparison: {exc}")

    report_path = os.path.join(output_dir, f"comparison_report_{timestamp}.md")
    generate_comparison_report(
        report_path,
        current_results_df,
        previous_results_df,
        timestamp,
        asset_names,
    )

    print("\nFinished.\n")


if __name__ == "__main__":
    main()
