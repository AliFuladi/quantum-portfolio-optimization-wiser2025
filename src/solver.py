# src/solver.py
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
import numpy as np


def find_best_penalty(qp: QuadraticProgram, penalty_range: np.ndarray) -> float | None:
    """Scan a list of penalty values and return the first one that gives a feasible solution."""
    try:
        numpy_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    except Exception:
        print("NumPyMinimumEigensolver not available; skipping penalty scan.")
        return None

    target_budget = int(qp.linear_constraints[0].rhs)

    for penalty in penalty_range:
        converter = QuadraticProgramToQubo(penalty=penalty)
        qubo = converter.convert(qp)
        result = numpy_solver.solve(qubo)
        if int(sum(result.x)) == target_budget:
            print(f"Feasible penalty found: {penalty}")
            return float(penalty)

    print("No feasible penalty found in the given range.")
    return None


def solve_with_classical_optimizer(qp: QuadraticProgram) -> np.ndarray:
    """Solve the model with NumPyMinimumEigensolver and return a binary solution vector."""
    try:
        numpy_solver = NumPyMinimumEigensolver()
        optimizer = MinimumEigenOptimizer(min_eigen_solver=numpy_solver)
        result = optimizer.solve(qp)
        return result.x
    except Exception as exc:
        print(f"Classical solver failed: {exc}")
        return np.zeros(qp.get_num_vars())


def solve_with_qaoa(qubo: QuadraticProgram) -> np.ndarray:
    """Solve the QUBO using a basic QAOA setup."""
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA())
    optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa_mes)
    result = optimizer.solve(qubo)
    return result.x
