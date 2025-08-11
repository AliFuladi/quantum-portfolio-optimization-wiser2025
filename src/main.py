# src/main.py
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Import all necessary modules
from src.utils import save_results, plot_portfolio_weights, plot_efficient_frontier, generate_comparison_report
from src.data_handler import fetch_stock_data, calculate_portfolio_metrics
from src.portfolio_model import create_quadratic_program
from src.solver import find_best_penalty, solve_with_classical_optimizer, solve_with_qaoa

# We need this specific converter for the QAOA part
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems import QuadraticProgram


def find_latest_results_file(directory, file_prefix):
    """
    Finds the latest file in a directory that matches a given prefix.

    Args:
        directory (str): The directory to search in.
        file_prefix (str): The prefix of the file name (e.g., 'solutions_').

    Returns:
        str: The full path to the latest file, or None if no file is found.
    """
    if not os.path.exists(directory):
        return None
    files = [os.path.join(directory, f) for f in os.listdir(
        directory) if f.startswith(file_prefix) and f.endswith('.csv')]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def main():
    """
    Main function to run the advanced end-to-end portfolio optimization project.
    """
    print("--- Advanced Portfolio Optimization Project ---")

    # Generate a unique timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Starting new run with timestamp: {timestamp}\n")

    # --- Configuration ---
    tickers = ['AAPL', 'AMZN', 'GOOG', 'KO', 'META',
               'MSFT', 'NFLX', 'NVDA', 'SBUX', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    num_assets_to_select = 5
    risk_factor = 0.5

    # --- Step 1: Data Handling ---
    print("1. Fetching and analyzing data...")
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    expected_returns, covariance_matrix, asset_names = calculate_portfolio_metrics(
        stock_data, risk_factor, num_assets_to_select)

    # --- Step 2: Classical Optimization ---
    print("\n2. Formulating and solving with a classical optimizer (NumPyMinimumEigensolver)...")
    qp = create_quadratic_program(
        expected_returns, covariance_matrix, num_assets_to_select)
    classical_solution = solve_with_classical_optimizer(qp)

    # --- Step 3: QAOA Optimization ---
    print("\n3. Solving with a quantum-inspired optimizer (QAOA)...")
    penalty_range = np.arange(10, 201, 10)
    best_penalty = find_best_penalty(qp, penalty_range)

    if best_penalty:
        qp_to_qubo = QuadraticProgramToQubo(penalty=best_penalty)
        qubo = qp_to_qubo.convert(qp)
        qaoa_solution = solve_with_qaoa(qubo)
    else:
        print("Could not find a feasible penalty factor. Skipping QAOA.")
        qaoa_solution = np.zeros(len(asset_names))

    # --- Step 4: Save and Plot Results ---
    print("\n4. Saving and plotting results...")
    # Plot the portfolio weights
    plot_portfolio_weights(qaoa_solution, classical_solution,
                           'weights_comparison', timestamp, asset_names)

    # Plot the efficient frontier, passing the classical and QAOA solutions
    # The plot_efficient_frontier function has been updated to accept these.
    plot_efficient_frontier(expected_returns, covariance_matrix,
                            classical_solution, qaoa_solution, 'efficient_frontier', timestamp)

    # Save the solutions to a CSV file
    results_data = {
        'Asset': asset_names,
        'Classical Solution': classical_solution,
        'QAOA Solution': qaoa_solution
    }
    save_results('solutions.csv', results_data, timestamp)

    # --- Step 5: Generate Comparison Report ---
    print("\n5. Generating comparison report...")
    output_dir = "results"
    all_solution_files = sorted([f for f in os.listdir(output_dir) if f.startswith(
        'solutions_') and f.endswith('.csv')], reverse=True)

    current_results_df = pd.DataFrame(results_data)
    previous_results_df = None

    if len(all_solution_files) > 1:
        latest_file_path = os.path.join(output_dir, all_solution_files[0])
        previous_file_path = os.path.join(output_dir, all_solution_files[1])
        try:
            previous_results_df = pd.read_csv(
                previous_file_path, encoding='utf-8')
            previous_results_df = previous_results_df[[
                'Asset', 'Classical Solution', 'QAOA Solution']]
        except Exception as e:
            print(f"Could not load previous results file for comparison: {e}")
            pass
    else:
        print("Skipping comparison report as this is the first run or only one run exists.")

    generate_comparison_report(os.path.join(output_dir, f"comparison_report_{timestamp}.md"),
                               current_results_df, previous_results_df, timestamp, asset_names)

    print("\nProject run finished successfully!")


if __name__ == '__main__':
    main()
