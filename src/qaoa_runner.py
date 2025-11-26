# src/qaoa_runner.py
import numpy as np
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


class QaoaRunner:
    """Small helper around QAOA + MinimumEigenOptimizer for a given QUBO."""

    def __init__(self, qubo: QuadraticProgram, num_assets: int):
        self.qubo = qubo
        self.num_assets = num_assets
        self.sampler = Sampler()

    def run(self) -> np.ndarray:
        """Run QAOA once and return the binary solution."""
        qaoa_mes = QAOA(optimizer=COBYLA(), sampler=self.sampler, reps=1)
        optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa_mes)
        result = optimizer.solve(self.qubo)

        solution = np.zeros(self.num_assets)
        for i, x in enumerate(result.x):
            solution[i] = x
        return solution

