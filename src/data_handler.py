# src/data_handler.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


def fetch_stock_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Download price history (Close) for the given tickers and date range."""
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    print("Data downloaded.")
    # Keep only the tickers we actually asked for
    return data[tickers]


def calculate_portfolio_metrics(
    data: pd.DataFrame,
    risk_factor: float,
    num_assets_limit: int,
):
    """
    Turn raw price data into:
    - a very simple expected-return estimate
    - an annualized covariance matrix
    - the list of tickers
    """
    asset_names = data.columns.tolist()
    print(f"Assets in this run: {asset_names}")
    print(f"Target number of names to pick: {num_assets_limit}")

    returns = data.pct_change().dropna()

    # Tiny toy model: predict tomorrow's return from yesterday's
    predicted_returns = np.zeros(returns.shape[1])
    for i in range(returns.shape[1]):
        stock_returns = returns.iloc[:, i].values.reshape(-1, 1)
        X = stock_returns[:-1]
        y = stock_returns[1:]

        model = LinearRegression()
        model.fit(X, y)

        last_return = X[-1].reshape(1, -1)
        predicted_next_return = model.predict(last_return)
        predicted_returns[i] = predicted_next_return * 252  # annualize

    # Still using historical covariance; not trying to be fancy here.
    covariance_matrix = returns.cov() * 252

    return predicted_returns, covariance_matrix, asset_names

