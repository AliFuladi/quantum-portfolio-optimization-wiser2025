# src/solver.py
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
import numpy as np


def find_best_penalty(qp: QuadraticProgram, penalty_range: np.array):
    """
    Finds the best penalty factor by checking for a feasible solution within a given range.

    This method iterates through a range of penalty values, solves the problem classically
    with each, and returns the first penalty value that yields a valid solution.

    Args:
        qp (QuadraticProgram): The formulated quadratic program.
        penalty_range (np.array): A range of penalty values to test.

    Returns:
        float: The best penalty value, or None if no feasible solution is found.
    """
    try:
        # We use a NumPyMinimumEigensolver to quickly test penalty values.
        # This is much faster than running the full QAOA algorithm for each test.
        numpy_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    except Exception as e:
        print(f"NumPyMinimumEigensolver is not available. Returning a default value.")
        return None

    for penalty in penalty_range:
        # In modern Qiskit, the penalty factor is passed when creating the converter.
        converter = QuadraticProgramToQubo(penalty=penalty)
        qubo = converter.convert(qp)
        # Solve with the classical solver to check feasibility.
        result = numpy_solver.solve(qubo)

        # Check if the solution satisfies the constraints of the original problem
        # The sum of selected assets must equal the number of assets to select.
        if int(sum(result.x)) == int(qp.linear_constraints[0].rhs):
            print(f"Found a feasible solution with penalty: {penalty}")
            return penalty

    print("Could not find a feasible penalty factor in the given range.")
    return None


def solve_with_classical_optimizer(qp: QuadraticProgram):
    """
    Solves the QUBO problem using a free classical optimizer (NumPyMinimumEigensolver).

    Args:
        qp (QuadraticProgram): The formulated QUBO problem.

    Returns:
        np.array: An array representing the solution.
    """
    try:
        # The NumPyMinimumEigensolver is a classical, exact solver for QUBO problems.
        # It is well-suited for small problem sizes.
        numpy_solver = NumPyMinimumEigensolver()
        # We initialize the MinimumEigenOptimizer with the numpy solver.
        optimizer = MinimumEigenOptimizer(min_eigen_solver=numpy_solver)
        result = optimizer.solve(qp)
        return result.x
    except Exception as e:
        print(f"Error solving classically with NumPyMinimumEigensolver: {e}")
        # Return an array of zeros if the classical solver fails.
        return np.zeros(qp.get_num_vars())


def solve_with_qaoa(qubo: QuadraticProgram):
    """
    Solves the QUBO problem using the QAOA algorithm.

    Args:
        qubo (QuadraticProgram): The QUBO problem.

    Returns:
        np.array: An array representing the solution.
    """
    # The QAOA algorithm is a MinimumEigenSolver, but it needs to be wrapped
    # in a MinimumEigenOptimizer to be able to call the solve() method.
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA())
    # Create the optimizer that will call the QAOA solver.
    optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa_mes)
    result = optimizer.solve(qubo)
    return result.x
