"""Volatility regime detection."""

from __future__ import annotations

import pandas as pd


def compute_rolling_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    return prices.pct_change().rolling(window=window).std() * (252 ** 0.5)


def classify_regime(volatility: float, low: float = 0.15, high: float = 0.35) -> str:
    if volatility < low:
        return "Low Volatility"
    if volatility < high:
        return "Medium Volatility"
    return "High Volatility"


def regime_color(regime: str) -> str:
    mapping = {
        "Low Volatility": "#2ecc71",
        "Medium Volatility": "#f39c12",
        "High Volatility": "#e74c3c",
    }
    return mapping.get(regime, "#95a5a6")


def regime_explanation(regime: str) -> str:
    explanations = {
        "Low Volatility": (
            "Markets are relatively calm. Lower risk of sudden drawdowns, "
            "but returns may be muted. Suitable for moderate position sizing."
        ),
        "Medium Volatility": (
            "Normal market conditions with balanced risk-return. "
            "Maintain diversified exposure and active monitoring."
        ),
        "High Volatility": (
            "Elevated price swings detected. Higher risk of sharp losses. "
            "Consider reducing leverage, tightening stop-losses, and hedging."
        ),
    }
    return explanations.get(regime, "Regime classification unavailable.")


def analyze_volatility_regime(prices: pd.Series) -> dict:
    vol_series = compute_rolling_volatility(prices)
    current_vol = float(vol_series.iloc[-1]) if not vol_series.empty else 0.0
    regime = classify_regime(current_vol)
    return {
        "Current Volatility": current_vol,
        "Regime": regime,
        "Color": regime_color(regime),
        "Explanation": regime_explanation(regime),
        "Volatility Series": vol_series,
    }
