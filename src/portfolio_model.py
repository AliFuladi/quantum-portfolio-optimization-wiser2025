# src/portfolio_model.py
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
import numpy as np


def create_quadratic_program(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    num_assets_to_select: int,
) -> QuadraticProgram:
    """Build Qiskit's QuadraticProgram for the portfolio-selection problem."""
    num_assets = len(expected_returns)
    qp = QuadraticProgram("portfolio_selection")

    # One binary variable per asset: 1 = picked, 0 = ignored
    for i in range(num_assets):
        qp.binary_var(name=f"x_{i}")

    # Budget: pick exactly `num_assets_to_select` names
    qp.linear_constraint(
        linear={f"x_{i}": 1 for i in range(num_assets)},
        sense="==",
        rhs=num_assets_to_select,
        name="budget",
    )

    # Minimize -return + risk (classic Markowitz style)
    linear_objective = -1 * expected_returns

    # covariance_matrix is a DataFrame here; use raw values
    quadratic_objective = covariance_matrix.values
    qp.minimize(linear=linear_objective, quadratic=quadratic_objective)
    return qp


def get_qubo(qp: QuadraticProgram, penalty: float) -> QuadraticProgram:
    """Convert the constrained problem into a QUBO using a simple penalty."""
    qp_to_qubo = QuadraticProgramToQubo(penalty=penalty)
    return qp_to_qubo.convert(qp)
