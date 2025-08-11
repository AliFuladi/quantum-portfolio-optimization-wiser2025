# src/portfolio_model.py
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
import numpy as np


def create_quadratic_program(expected_returns: np.array, covariance_matrix: np.array, num_assets_to_select: int):
    """
    Creates a Quadratic Program formulation of the portfolio optimization problem.

    Args:
        expected_returns (np.array): The expected returns for each asset.
        covariance_matrix (np.array): The covariance matrix of the assets.
        num_assets_to_select (int): The number of assets to select in the final portfolio.

    Returns:
        qiskit_optimization.QuadraticProgram: The formulated quadratic program.
    """
    num_assets = len(expected_returns)
    qp = QuadraticProgram("Portfolio Optimization")

    # Create binary variables for each asset (1 if selected, 0 if not)
    for i in range(num_assets):
        qp.binary_var(name=f'x_{i}')

    # Add the budget constraint: sum of selected assets must equal num_assets_to_select
    qp.linear_constraint(
        linear={f'x_{i}': 1 for i in range(num_assets)},
        sense="==",
        rhs=num_assets_to_select,
        name="budget_constraint"
    )

    # Set the objective function: Maximize return and minimize risk
    # This is a Quadratic Unconstrained Binary Optimization (QUBO) problem
    # The objective is to maximize the expected return and minimize the portfolio risk.
    # We formulate this as minimizing -return + risk.

    # Linear part: Minimize the negative of the expected return
    linear_objective = -1 * expected_returns

    # Quadratic part: Minimize the risk (covariance matrix)
    # The error was caused here because the input was a pandas DataFrame.
    # We must pass a simple numpy array. We use .values to extract the numerical data.
    quadratic_objective = covariance_matrix.values
    qp.minimize(linear=linear_objective, quadratic=quadratic_objective)

    return qp


def get_qubo(qp: QuadraticProgram, penalty: float):
    """
    Converts a Quadratic Program into a Quadratic Unconstrained Binary Optimization (QUBO) problem.

    Args:
        qp (qiskit_optimization.QuadraticProgram): The formulated quadratic program.
        penalty (float): The penalty factor for the budget constraint.

    Returns:
        qiskit_optimization.QuadraticProgram: The QUBO problem.
    """
    # The penalty method is a way to convert a constrained problem into an unconstrained one.
    # We add a term to the objective function that penalizes solutions that violate the constraint.
    # Qiskit's `QuadraticProgramToQubo` converter can handle this automatically
    # if we provide a `penalty`.
    qp_to_qubo = QuadraticProgramToQubo(penalty=penalty)
    qubo = qp_to_qubo.convert(qp)
    return qubo
