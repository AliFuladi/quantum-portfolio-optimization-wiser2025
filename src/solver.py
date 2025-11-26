# src/solver.py
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
import numpy as np


def find_best_penalty(qp: QuadraticProgram, penalty_range: np.ndarray):
    """Scan a list of penalty values and return the first that yields a feasible solution."""
    try:
        classical_mes = NumPyMinimumEigensolver()
        classical_optimizer = MinimumEigenOptimizer(min_eigen_solver=classical_mes)
    except Exception as exc:
        print(f"NumPyMinimumEigensolver not available, skipping penalty search: {exc}")
        return None

    # Try to locate the budget constraint explicitly
    target_budget = None
    for lc in qp.linear_constraints:
        if lc.name == "budget":
            target_budget = int(lc.rhs)
            break

    if target_budget is None:
        # Fall back to the first constraint if something odd happens
        if qp.linear_constraints:
            target_budget = int(qp.linear_constraints[0].rhs)
        else:
            print("No linear constraints found; nothing to penalise.")
            return None

    for penalty in penalty_range:
        converter = QuadraticProgramToQubo(penalty=float(penalty))
        qubo = converter.convert(qp)
        result = classical_optimizer.solve(qubo)

        # Enforce the budget constraint by looking at the number of selected assets
        if int(np.round(np.sum(result.x))) == target_budget:
            print(f"Feasible penalty found: {penalty}")
            return float(penalty)

    print("No feasible penalty found in the given range.")
    return None


def solve_with_classical_optimizer(qp: QuadraticProgram) -> np.ndarray:
    """Solve the model with a classical eigensolver and return the binary solution."""
    try:
        mes = NumPyMinimumEigensolver()
        optimizer = MinimumEigenOptimizer(min_eigen_solver=mes)
        result = optimizer.solve(qp)
        return result.x
    except Exception as exc:
        print(f"Classical solver failed: {exc}")
        # If this blows up, it's better to see it than silently hide it
        raise


def solve_with_qaoa(qubo: QuadraticProgram) -> np.ndarray:
    """Solve the QUBO instance using a small QAOA setup."""
    # Small, reasonably cheap QAOA configuration for this toy problem
    qaoa_mes = QAOA(
        sampler=Sampler(),
        optimizer=COBYLA(),
    )
    optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa_mes)
    result = optimizer.solve(qubo)
    return result.x
