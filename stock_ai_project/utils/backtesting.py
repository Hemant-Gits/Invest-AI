"""Backtesting and strategy evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_strategy(prices: pd.Series, signals: pd.Series) -> pd.DataFrame:
    """Simple long-only strategy simulation from binary signals."""
    returns = prices.pct_change().fillna(0)
    position = signals.shift(1).fillna(0)
    strategy_returns = returns * position
    equity = (1 + strategy_returns).cumprod()
    return pd.DataFrame({
        "Date": prices.index,
        "Price": prices.values,
        "Signal": signals.values,
        "Strategy_Return": strategy_returns.values,
        "Equity_Curve": equity.values,
    })


def max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min() * 100)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.05) -> float:
    if returns.std() == 0:
        return 0.0
    excess = returns.mean() * 252 - risk_free
    return float(excess / (returns.std() * np.sqrt(252)))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.05) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    excess = returns.mean() * 252 - risk_free
    return float(excess / (downside.std() * np.sqrt(252)))


def win_rate(signals: pd.Series, prices: pd.Series) -> float:
    returns = prices.pct_change()
    trades = returns[signals.shift(1) == 1].dropna()
    if trades.empty:
        return 0.0
    return float((trades > 0).mean() * 100)


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float(((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100)


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0] - 1) * 100)


def risk_recommendation(sharpe: float, max_dd: float, win: float) -> str:
    if sharpe >= 1.2 and max_dd > -15 and win >= 55:
        return "Low Risk"
    if sharpe >= 0.6 and max_dd > -25:
        return "Medium Risk"
    return "High Risk"


def run_backtest_report(prices: pd.Series, predicted_prices: pd.Series) -> dict:
    """Generate backtest metrics from actual vs predicted prices."""
    aligned = pd.DataFrame({"actual": prices, "predicted": predicted_prices}).dropna()
    if len(aligned) < 5:
        return {}

    signals = (aligned["predicted"].diff() > 0).astype(int)
    sim = simulate_strategy(aligned["actual"], signals)
    rets = sim["Strategy_Return"]

    equity = sim["Equity_Curve"]
    metrics = {
        "Sharpe Ratio": sharpe_ratio(rets),
        "Sortino Ratio": sortino_ratio(rets),
        "Max Drawdown": max_drawdown(equity),
        "Win Rate": win_rate(signals, aligned["actual"]),
        "CAGR": cagr(equity),
        "Total Return": total_return(equity),
    }
    metrics["Risk Recommendation"] = risk_recommendation(
        metrics["Sharpe Ratio"], metrics["Max Drawdown"], metrics["Win Rate"]
    )
    metrics["Equity_Curve"] = sim
    return metrics
