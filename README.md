# Hybrid Quantum-Classical Portfolio Optimization

This project demonstrates a modern approach to portfolio optimization by combining traditional machine learning with quantum computing concepts.

It uses an AI model to predict stock returns and then employs both a classical optimizer and a quantum-inspired algorithm (QAOA) to select the most efficient portfolio.

## How It Works

1.  **AI-Driven Data:** It fetches real-world stock data and uses a simple AI model (Linear Regression) to predict future returns. This is a crucial step for making more informed investment decisions.
2.  **Problem Formulation:** The portfolio selection problem is converted into a mathematical format suitable for both classical and quantum solvers.
3.  **Classical Solution:** A standard classical optimizer finds the ideal portfolio. This serves as a benchmark for our quantum solution.
4.  **Quantum-Inspired Solution (QAOA):** A QAOA algorithm is used to find a near-optimal solution. This showcases how quantum computers can tackle complex optimization problems.
5.  **Comparison:** The project compares the two solutions to see how they differ.

## Project Structure

```
