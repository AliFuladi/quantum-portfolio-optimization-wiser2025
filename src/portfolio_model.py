# src/portfolio_model.py
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
import numpy as np


def create_quadratic_program(
    expected_returns: np.ndarray,
    covariance_matrix,
    num_assets_to_select: int,
    risk_factor: float,
) -> QuadraticProgram:
    """Build the QuadraticProgram for the portfolio-selection toy model."""
    num_assets = len(expected_returns)
    qp = QuadraticProgram("portfolio_selection")

    # One binary variable per asset: 1 = picked, 0 = ignored
    for i in range(num_assets):
        qp.binary_var(name=f"x_{i}")

    # Budget constraint: pick exactly `num_assets_to_select` assets
    qp.linear_constraint(
        linear={f"x_{i}": 1 for i in range(num_assets)},
        sense="==",
        rhs=num_assets_to_select,
        name="budget",
    )

    # Objective: minimise -return + risk_factor * risk
    linear_objective = -1.0 * expected_returns

    if hasattr(covariance_matrix, "values"):
        cov_values = covariance_matrix.values
    else:
        cov_values = np.array(covariance_matrix)

    quadratic_objective = risk_factor * cov_values

    qp.minimize(linear=linear_objective, quadratic=quadratic_objective)
    return qp


def get_qubo(qp: QuadraticProgram, penalty: float) -> QuadraticProgram:
    """Convert the constrained model into a QUBO using a penalty method."""
    converter = QuadraticProgramToQubo(penalty=penalty)
    return converter.convert(qp)
