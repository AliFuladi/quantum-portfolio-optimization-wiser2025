# Hybrid Quantum–Classical Portfolio Optimization

This repository contains a small, end-to-end experiment in hybrid quantum–classical portfolio optimization.

The idea is simple: take a handful of real stocks, build a toy Markowitz-style model, encode it as a QUBO, and solve it in two different ways:

- with a classical eigensolver, and  
- with QAOA implemented in Qiskit.

The goal is **not** to beat real-world trading desks. The point is to have a concrete, reproducible example of how a quantum algorithm can sit next to a classical workflow and be compared against a known baseline.

---

## 1. Background and Approach

Modern Portfolio Theory (Markowitz) tells us that, under some assumptions, we can trade off expected return against risk (variance) and find “efficient” portfolios along a frontier.

For a small number of assets this is easy to handle classically, but the number of possible 0/1 selection vectors grows exponentially with the number of assets. That combinatorial structure is what makes this problem interesting for quantum algorithms.

In this project I do the following:

1. **Data**  
   - Pull daily price data for 10 large-cap stocks from Yahoo Finance (2020–2024).  
   - Compute daily returns and annualise them.

2. **Toy return & risk model**  
   - For each asset, fit a tiny 1D linear regression that predicts “tomorrow’s” return from “yesterday’s” return and use that as a crude expected-return estimate.  
   - Build the historical covariance matrix of returns and annualise it.

3. **Optimization model (Markowitz-style)**  
   - Binary variable \(x_i \in \{0, 1\}\) for each asset \(i\):  
     - \(x_i = 1\): asset is in the portfolio,  
     - \(x_i = 0\): asset is ignored.  
   - Hard constraint: pick exactly `k` assets (budget on number of names, not money).  
   - Objective: minimise  
     \[
       - \text{expected\_return}(x) + \lambda \cdot \text{risk}(x)
     \]  
     where \(\lambda\) is a user-chosen `risk_factor`.

4. **From constrained QP to QUBO**  
   - Build a `QuadraticProgram` with `qiskit-optimization`.  
   - Use `QuadraticProgramToQubo` with a penalty parameter to fold the budget constraint into the objective and obtain a QUBO instance.

5. **Solvers**  
   - **Classical baseline**:  
     - `NumPyMinimumEigensolver` + `MinimumEigenOptimizer`.  
     - This gives a “ground truth” solution for such a small problem.
   - **QAOA**:  
     - `QAOA` with `Sampler` (Aer backend) and `COBYLA` as the classical optimizer.  
     - Run on the same QUBO and compare the selected assets with the classical solution.

6. **Metrics and visualisation**  
   - For each solution (classical and QAOA), compute:  
     - expected portfolio return,  
     - portfolio volatility (risk),  
     using weights proportional to the selected assets.  
   - Generate:
     - a bar chart comparing portfolio weights,  
     - a “toy” efficient frontier with random portfolios plus the two solutions overlaid.

On this small instance, QAOA often finds exactly the same portfolio as the classical solver, and sometimes a slightly different feasible portfolio with similar return–risk trade-offs. That is roughly what you expect from a variational quantum algorithm running on a simulator.

---

## 2. Team

- **Member:** Ali Fuladi  
- **Program:** WISER & Womanium Quantum Program 2025  
- **WISER ID:** `gst-qoULCjoaMQy5jWe`

---

## 3. Repository Structure

```text
.
├─ src/
│  ├─ __init__.py
│  ├─ main.py             # Entry point, orchestrates the full pipeline
│  ├─ data_handler.py     # Data download + return/covariance estimation
│  ├─ portfolio_model.py  # Build QuadraticProgram and convert to QUBO
│  ├─ solver.py           # Classical solver, QAOA solver, penalty search
│  └─ utils.py            # Saving results, plots, tiny reporting helpers
└─ results/               # Created at runtime; CSV, plots, markdown reports
