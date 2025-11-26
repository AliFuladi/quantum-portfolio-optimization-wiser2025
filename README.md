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

The `results/` folder is created automatically on the first run.

```

---

## 4. Environment and Dependencies

This project is intended to run with **Python 3.11.x**.

Install dependencies with:

```bash
pip install -r requirements.txt
```

requirements.txt:

```text
# Tested with Python 3.11.x
numpy==1.24.4
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.2

qiskit==0.45.1
qiskit-algorithms==0.2.1
qiskit-optimization==0.6.0
qiskit-aer==0.12.2

yfinance==0.2.65
```

A typical workflow is:

```bash
python -m venv .qpowvenv
.\.qpowvenv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. How to Run

From the repository root:

```bash
python -m src.main
```

The script will:

1. Download historical daily prices for the selected tickers.
2. Compute expected returns and the annualised covariance matrix.
3. Build the quadratic optimization model with a budget constraint.
4. Solve it once with a classical eigensolver.
5. Convert the model to a QUBO, search for a reasonable penalty, and run QAOA on it.
6. Compute portfolio-level metrics for both solutions.
7. Save CSV results, plots, and a small markdown comparison report in the `results/` folder.

---

## 6. Outputs

Each run creates time-stamped files in `results/`:

- `solutions_<timestamp>.csv`
   - Asset names
   - Binary selection vectors for the classical and QAOA solvers

- `weights_comparison_<timestamp>.png`
   - Bar chart comparing normalised portfolio weights
   - One bar per asset for the classical and QAOA portfolios

- `efficient_frontier_<timestamp>.png`
   - Scatter plot of random portfolios in (risk, return) space
   - Marked points for the classical and QAOA portfolios

- `comparison_report_<timestamp>.md`
   - Short markdown report listing which assets were selected
   - If a previous run exists, shows how the selections changed between runs

---

## 7. What the Results Show

On this small test case:

- The classical solver reliably finds a portfolio on the upper part of the sampled efficient frontier and does so very quickly.

- QAOA:
   - Often matches the classical solution exactly (same assets, same risk/return).
   - Sometimes returns a slightly different feasible portfolio with a very similar risk–return profile.

- This is consistent with the behaviour of variational quantum algorithms on NISQ-style simulators: they can approximate good solutions to combinatorial problems and, with the right encoding and penalties, stay close to the classical optimum.

This project is deliberately modest in scope: it is a proof of concept that shows how to wire together:

1. real market data,
2. a simple Markowitz-style model,
3. QUBO encoding,
4. a hybrid quantum–classical solver pipeline.

From here, natural next steps would be to experiment with more realistic constraints, different cost functions, and larger universes of assets as quantum hardware and algorithms improve.

```makefile
::contentReference[oaicite:0]{index=0}
```

