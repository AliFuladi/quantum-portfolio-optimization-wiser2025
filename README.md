# Hybrid Quantum-Classical Portfolio Optimization

## Introduction
The process of portfolio optimization is a cornerstone of modern financial theory, seeking to build a portfolio of assets that maximizes expected returns for a given level of risk. This problem, first formally addressed by Harry Markowitz in his seminal work on Modern Portfolio Theory (MPT), established the principle of diversification—that a combination of assets can yield a higher risk-adjusted return than any single asset. The core challenge lies in the immense number of possible combinations. As the number of available assets grows, the number of potential portfolios grows exponentially, making a brute-force search for the optimal solution computationally intractable for even modestly sized problems. This is a classic example of a complex optimization problem, and it's precisely the kind of challenge that the burgeoning field of quantum computing aims to address.

Classical methods, such as quadratic programming and other convex optimization techniques, have been highly effective for this problem, especially for relatively small portfolios. However, these methods often rely on certain assumptions, such as quadratic utility functions or perfectly liquid markets, which may not always hold true. Furthermore, they become computationally expensive and time-consuming when the number of assets, constraints, or complex dependencies increases. This limitation has spurred research into alternative computational paradigms that can scale more effectively.

This project delves into one such alternative: quantum computing. Unlike classical computers that store information in bits (0s or 1s), quantum computers use qubits, which can exist in a superposition of both states simultaneously. This unique property, along with quantum entanglement, allows quantum computers to process information in fundamentally different ways. While the technology is still in the Noisy Intermediate-Scale Quantum (NISQ) era, characterized by limited qubit counts and a high degree of error, it is a powerful testbed for developing and testing algorithms that could one day outperform their classical counterparts.

The solution presented here leverages the Quantum Approximate Optimization Algorithm (QAOA), a prime example of a hybrid quantum-classical algorithm. In this paradigm, a classical computer handles a significant portion of the workload by optimizing a set of parameters, while a quantum computer executes the most computationally intensive part: generating and measuring quantum states. QAOA is specifically designed to solve combinatorial optimization problems by transforming them into a quantum mechanical framework. The algorithm works by iteratively applying two key quantum operations, known as the cost Hamiltonian and the mixer Hamiltonian. The cost Hamiltonian encodes the optimization problem's objective function (in our case, maximizing return and minimizing risk) into the quantum circuit. The mixer Hamiltonian then drives the system through different states, allowing it to explore the solution space. A classical optimizer, running on a standard computer, then adjusts the parameters of these Hamiltonians to guide the quantum computer toward a better solution. This cyclical process of quantum state preparation and classical parameter optimization continues until a satisfactory or optimal solution is found.

To apply QAOA to our portfolio problem, we first need to reframe it as a Quadratic Unconstrained Binary Optimization (QUBO) problem. A QUBO is a mathematical model for a wide range of optimization challenges, where the objective is to minimize a quadratic function of binary variables. In our context, we can define a binary variable, x_i, for each asset i, where x_i=1 if the asset is selected for the portfolio and x_i=0 if it is not. We can then encode the financial objective—the trade-off between risk and return—into the coefficients of the QUBO matrix. The constraints of the problem, such as a fixed budget or a set number of assets to select, are "penalized" and integrated into the objective function itself, effectively making the problem "unconstrained" in the quantum solver's view. This structured representation makes the problem directly amenable to being solved by QAOA.

The final and crucial step of this project is the comparison of the QAOA solution with a classical one. While a classical exact solver (like the one used here) can easily and quickly find the provably optimal solution for the small number of assets in this example, it serves as a vital benchmark. By comparing the results, we can validate the output of the QAOA algorithm, assess its performance, and demonstrate its potential. This comparison is not just about finding the "best" answer, but about proving the viability and potential of quantum-inspired methods for finance, paving the way for a future where such algorithms can tackle problems that are beyond the reach of today's supercomputers. This project, therefore, is a proof of concept, demonstrating a concrete application of quantum algorithms to a real-world financial challenge.

---

## Team Information
- **Member:** Ali Fuladi — WISER ID: gst-qoULCjoaMQy5jWe

---

## Project Structure
- **`src/main.py`** — Entry point; orchestrates all steps from data fetching to solving and saving results.  
- **`src/data_handler.py`** — Fetches historical stock data from Yahoo Finance and calculates returns and covariance matrix.  
- **`src/portfolio_model.py`** — Formulates the problem as a Quadratic Program and converts to QUBO.  
- **`src/solver.py`** — Finds penalty factor and runs classical & QAOA solvers.  
- **`src/qaoa_runner.py`** — QAOA execution wrapper with Qiskit.  
- **`src/utils.py`** — Helper functions for saving CSV results and generating plots.

---

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Contents of requirements.txt:
```bash
numpy==1.24.4
scipy==1.10.1
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.2
qiskit==0.45.1
qiskit-algorithms==0.2.1
qiskit-optimization==0.6.0
qiskit-aer==0.12.2
docplex==2.25.236
yfinance==0.2.65
tabulate==0.9.0
gurobipy==12.0.3
```

## How to Run
```bash
python -m src.main
```
