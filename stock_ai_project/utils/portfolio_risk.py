"""Portfolio risk analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_returns(symbols: list[str], period: str = "1y") -> pd.DataFrame:
    """Download daily returns for multiple tickers."""
    frames = {}
    for symbol in symbols:
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist.empty:
            continue
        frames[symbol] = hist["Close"].pct_change()
    if not frames:
        return pd.DataFrame()
    returns = pd.DataFrame(frames).dropna()
    return returns


def portfolio_metrics(symbols: list[str], weights: list[float] | None = None) -> dict:
    """Compute portfolio return, volatility, correlation, diversification."""
    returns = fetch_returns(symbols)
    if returns.empty:
        return {}

    n = len(returns.columns)
    if weights is None:
        weights = [1 / n] * n
    weights = np.array(weights[:n])
    weights = weights / weights.sum()

    port_returns = (returns * weights).sum(axis=1)
    corr = returns.corr()

    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    diversification_score = float(max(0, (1 - avg_corr) * 100))

    return {
        "Portfolio Return": float(port_returns.mean() * 252 * 100),
        "Portfolio Volatility": float(port_returns.std() * np.sqrt(252) * 100),
        "Correlation Matrix": corr,
        "Diversification Score": diversification_score,
        "Weights": dict(zip(returns.columns.tolist(), weights.tolist())),
        "Returns": returns,
    }
