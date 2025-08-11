# src/qaoa_runner.py
import numpy as np
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


class QaoaRunner:
    """
    A class to run the QAOA algorithm on a given QUBO problem.
    This version uses the modern Qiskit Sampler primitive (v1.0+ compatible).
    """

    def __init__(self, qubo: QuadraticProgram, num_assets: int):
        self.qubo = qubo
        self.num_assets = num_assets
        # Use the modern Sampler primitive from qiskit_aer
        self.sampler = Sampler()

    def run(self):
        """
        Executes the QAOA algorithm to solve the QUBO.

        Returns:
            np.array: The solution as a binary array.
        """
        optimizer = COBYLA()

        # The QAOA solver
        qaoa_mes = QAOA(
            optimizer=optimizer,
            sampler=self.sampler,
            reps=1
        )

        # Use MinimumEigenOptimizer to solve the QUBO problem
        optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa_mes)
        result = optimizer.solve(self.qubo)

        # The result.x contains the solution in a binary array format
        # We need to make sure the solution has the correct length
        solution = np.zeros(self.num_assets)
        for i in range(len(result.x)):
            solution[i] = result.x[i]

        return solution
