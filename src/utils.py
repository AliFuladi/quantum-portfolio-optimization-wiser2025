# src/utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def save_results(file_name: str, data: dict, timestamp: str, output_dir="results"):
    """
    Saves optimization results to a CSV file with a timestamp.

    Args:
        file_name (str): The base name of the file to save the data to.
        data (dict): A dictionary containing the results.
        timestamp (str): The timestamp to append to the file name.
        output_dir (str): The directory where the results will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_file_name = f"{os.path.splitext(file_name)[0]}_{timestamp}.csv"

    df = pd.DataFrame(data)
    file_path = os.path.join(output_dir, final_file_name)
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")


def plot_portfolio_weights(qaoa_solution: np.array, classical_solution: np.array, file_name: str, timestamp: str, asset_names: list, output_dir="results"):
    """
    Creates and saves a bar chart comparing the two solutions with a timestamp.

    Args:
        qaoa_solution (np.array): The solution from the QAOA algorithm.
        classical_solution (np.array): The solution from the classical optimizer.
        file_name (str): The base name of the file to save the plot to.
        timestamp (str): The timestamp to append to the file name.
        asset_names (list): List of asset names.
        output_dir (str): The directory where the plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_file_name = f"{os.path.splitext(file_name)[0]}_{timestamp}.png"
    file_path = os.path.join(output_dir, final_file_name)

    # Convert binary solutions to portfolio weights
    qaoa_weights = qaoa_solution / \
        np.sum(qaoa_solution) if np.sum(
            qaoa_solution) > 0 else np.zeros(len(qaoa_solution))
    classical_weights = classical_solution / np.sum(classical_solution) if np.sum(
        classical_solution) > 0 else np.zeros(len(classical_solution))

    df = pd.DataFrame({
        'Asset': asset_names,
        'QAOA Solution': qaoa_weights,
        'Classical Solution': classical_weights
    }).set_index('Asset')

    df.plot(kind='bar', figsize=(12, 7))
    plt.title(f'Portfolio Weights Comparison ({timestamp})')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Portfolio weights plot saved to {file_path}")


def plot_efficient_frontier(expected_returns: np.array, covariance_matrix: np.array, classical_solution: np.array, qaoa_solution: np.array, file_name: str, timestamp: str, output_dir="results"):
    """
    Plots the efficient frontier for the given portfolio data, and marks the classical and QAOA solutions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_file_name = f"{os.path.splitext(file_name)[0]}_{timestamp}.png"
    file_path = os.path.join(output_dir, final_file_name)

    num_assets = len(expected_returns)
    num_portfolios = 5000

    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.sum(expected_returns * weights)
        portfolio_std_dev = np.sqrt(
            np.dot(weights.T, np.dot(covariance_matrix, weights)))

        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = portfolio_return / portfolio_std_dev  # Sharpe Ratio

    plt.figure(figsize=(12, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :],
                cmap='viridis', s=20, label='Random Portfolios')
    plt.colorbar(label='Sharpe Ratio')

    # Calculate and plot the classical solution's point
    classical_return, classical_risk = calculate_portfolio_metrics(
        classical_solution, expected_returns, covariance_matrix)
    plt.scatter(classical_risk, classical_return, marker='*',
                s=200, color='red', label='Classical Solution')

    # Calculate and plot the QAOA solution's point, now with a blue star
    qaoa_return, qaoa_risk = calculate_portfolio_metrics(
        qaoa_solution, expected_returns, covariance_matrix)
    plt.scatter(qaoa_risk, qaoa_return, marker='*', s=200,
                color='blue', label='QAOA Solution')

    plt.title(f'Efficient Frontier ({timestamp})')
    plt.xlabel('Expected Volatility (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Efficient frontier plot saved to {file_path}")


def generate_comparison_report(report_file_path: str, current_results: pd.DataFrame, previous_results: pd.DataFrame, timestamp: str, asset_names: list):
    """
    Generates a Markdown report comparing the latest run with the previous one.
    """
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(f"# Portfolio Optimization Run Report - {timestamp}\n\n")
        f.write("## 1. Solutions\n")
        f.write("### Portfolio Weights\n")
        f.write(
            "Here is a comparison of the selected assets by the Classical and QAOA solvers.\n\n")

        # Use markdown table for solutions
        f.write("| Asset | Classical Solution | QAOA Solution |\n")
        f.write("|-------|--------------------|---------------|\n")
        for idx, asset in enumerate(asset_names):
            classical_val = '✅' if current_results['Classical Solution'][idx] == 1 else '❌'
            qaoa_val = '✅' if current_results['QAOA Solution'][idx] == 1 else '❌'
            f.write(f"| {asset} | {classical_val} | {qaoa_val} |\n")
        f.write("\n")

        if previous_results is not None:
            f.write("## 2. Comparison with Previous Run\n")
            f.write(
                "This section compares the current run's results with the most recent previous run.\n\n")

            # Convert solutions to a more readable format for the report
            current_classical_assets = ', '.join([asset_names[i] for i, x in enumerate(
                current_results['Classical Solution']) if x == 1])
            previous_classical_assets = ', '.join([asset_names[i] for i, x in enumerate(
                previous_results['Classical Solution']) if x == 1])

            current_qaoa_assets = ', '.join([asset_names[i] for i, x in enumerate(
                current_results['QAOA Solution']) if x == 1])
            previous_qaoa_assets = ', '.join([asset_names[i] for i, x in enumerate(
                previous_results['QAOA Solution']) if x == 1])

            f.write("### Classical Solution Change\n")
            f.write(f"- **Current Run:** {current_classical_assets}\n")
            f.write(f"- **Previous Run:** {previous_classical_assets}\n\n")

            f.write("### QAOA Solution Change\n")
            f.write(f"- **Current Run:** {current_qaoa_assets}\n")
            f.write(f"- **Previous Run:** {previous_qaoa_assets}\n\n")

        else:
            f.write("## 2. Comparison with Previous Run\n")
            f.write("No previous run data found for comparison.\n\n")

    print(f"Comparison report generated and saved to {report_file_path}")


def calculate_portfolio_metrics(solution: np.array, expected_returns: np.array, covariance_matrix: np.array):
    """
    Calculates the expected return and risk (volatility) of a portfolio based on a given solution.

    Args:
        solution (np.array): A binary array where 1 means the asset is selected, 0 otherwise.
        expected_returns (np.array): The expected returns for each asset.
        covariance_matrix (np.array): The covariance matrix of the assets.

    Returns:
        tuple: A tuple containing (portfolio_return, portfolio_risk).
    """
    selected_assets = solution
    if np.sum(selected_assets) == 0:
        return 0, 0

    # For selected assets, normalize the weights to sum to 1
    weights = selected_assets / np.sum(selected_assets)

    # Calculate portfolio return
    portfolio_return = np.dot(weights, expected_returns)

    # Calculate portfolio risk (standard deviation)
    portfolio_risk = np.sqrt(
        np.dot(weights.T, np.dot(covariance_matrix, weights)))

    return portfolio_return, portfolio_risk
