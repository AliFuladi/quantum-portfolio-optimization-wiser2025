# src/data_handler.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


def fetch_stock_data(tickers: list, start_date: str, end_date: str):
    """
    Fetches historical stock price data from Yahoo Finance without caching.

    Args:
        tickers (list): A list of stock tickers (e.g., ['AAPL', 'GOOG']).
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame of adjusted closing prices.
    """
    print(
        f"Fetching new data for {tickers} from {start_date} to {end_date}...")
    # The yf.download function now defaults to auto_adjust=True.
    # It also no longer returns a MultiIndex DataFrame with 'Adj Close'
    # when multiple tickers are requested. The adjusted prices are
    # returned directly under the 'Close' column.
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    print("Data fetched successfully.")

    # Ensure the data only contains the tickers we requested
    data = data[tickers]
    return data


def calculate_portfolio_metrics(data: pd.DataFrame, risk_factor: float, num_assets_limit: int):
    """
    Calculates portfolio metrics and prepares data for the optimization model.

    Args:
        data (pd.DataFrame): DataFrame of historical stock prices.
        risk_factor (float): A factor for the risk component in the objective function.
        num_assets_limit (int): The number of assets to select.

    Returns:
        tuple: A tuple containing:
            - np.array: The predicted expected returns.
            - pd.DataFrame: The annualized covariance matrix.
            - list: The list of asset names (tickers).
    """
    asset_names = data.columns.tolist()
    print(f"Available assets: {asset_names}")
    print(
        f"Note: The AI model will be configured to select up to {num_assets_limit} assets.")

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # -----------------------------------------------------------
    # AI Prediction Section
    # -----------------------------------------------------------

    # In this simplified example, we use a Linear Regression model to predict
    # the next day's return based on the previous day's return.
    predicted_returns = np.zeros(returns.shape[1])
    for i in range(returns.shape[1]):
        stock_returns = returns.iloc[:, i].values.reshape(-1, 1)
        X = stock_returns[:-1]
        y = stock_returns[1:]
        # Train a simple Linear Regression model
        model = LinearRegression()
        model.fit(X, y)
        # Predict the next return based on the last known return
        last_return = X[-1].reshape(1, -1)
        predicted_next_return = model.predict(last_return)
        # We annualize this single predicted return
        predicted_returns[i] = predicted_next_return * 252

    # The covariance matrix is still calculated from historical data for simplicity.
    # A more advanced AI model (e.g., LSTM) could also predict this.
    covariance_matrix = returns.cov() * 252

    # Return the metrics needed for the optimization model
    return predicted_returns, covariance_matrix, asset_names
