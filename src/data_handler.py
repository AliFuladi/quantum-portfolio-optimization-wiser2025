# src/data_handler.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def fetch_stock_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers and date range."""
    print(f"Downloading data for {tickers} from {start_date} to {end_date} ...")
    raw = yf.download(tickers, start=start_date, end=end_date)

    # On recent yfinance versions "Close" already contains (adjusted) prices
    if isinstance(raw, pd.DataFrame) and "Close" in raw.columns:
        data = raw["Close"]
    else:
        # If the structure is different, fall back to the raw frame
        data = raw

    # Keep a consistent column order and drop tickers that failed to download
    existing = [t for t in tickers if t in data.columns]
    data = data.loc[:, existing]

    print(f"Downloaded price history with shape {data.shape}.")
    return data


def calculate_portfolio_metrics(
    data: pd.DataFrame,
    num_assets_limit: int,
):
    """Turn price data into expected returns, covariance matrix and asset names."""
    asset_names = list(data.columns)
    print(f"Assets in this run: {asset_names}")
    print(f"Budget (number of names to pick): {num_assets_limit}")

    # Daily returns
    returns = data.pct_change().dropna()

    # Tiny toy model:
    # predict "tomorrow's" return from "yesterday's" return with a 1D linear regression
    predicted_returns = np.zeros(len(asset_names))
    for idx in range(len(asset_names)):
        series = returns.iloc[:, idx].values.reshape(-1, 1)
        if len(series) < 2:
            predicted_returns[idx] = 0.0
            continue

        X = series[:-1]
        y = series[1:]

        model = LinearRegression()
        model.fit(X, y)

        last_value = X[-1].reshape(1, -1)
        predicted_next = model.predict(last_value)[0, 0]
        # annualise from a one-step forecast
        predicted_returns[idx] = float(predicted_next * 252)

    # Historical covariance, annualised
    covariance_matrix = returns.cov() * 252

    return predicted_returns, covariance_matrix, asset_names
