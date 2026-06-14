"""Feature engineering for stock prediction models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.config import FEATURE_COLUMNS


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def engineer_features(df: pd.DataFrame, sentiment_score: float | None = None) -> pd.DataFrame:
    """Create ML feature matrix from OHLCV data."""
    if df.empty:
        return pd.DataFrame()

    data = df.copy()
    if "Date" in data.columns:
        data = data.set_index("Date")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in data.columns:
            return pd.DataFrame()

    for lag in [1, 3, 5, 10]:
        data[f"Lag_{lag}"] = data["Close"].shift(lag)

    for window in [5, 10, 20]:
        data[f"Rolling_Mean_{window}"] = data["Close"].rolling(window=window).mean()

    data["RSI"] = compute_rsi(data["Close"])
    data["MACD"], data["MACD_Signal"] = compute_macd(data["Close"])

    bb_mid = data["Close"].rolling(window=20).mean()
    bb_std = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = bb_mid + 2 * bb_std
    data["BB_Lower"] = bb_mid - 2 * bb_std
    data["Volatility"] = data["Close"].pct_change().rolling(window=20).std()

    if sentiment_score is not None:
        data["Sentiment_Score"] = sentiment_score

    feature_cols = [c for c in FEATURE_COLUMNS if c in data.columns]
    if sentiment_score is not None:
        feature_cols.append("Sentiment_Score")

    engineered = data[feature_cols].copy()
    engineered["Target"] = data["Close"].shift(-1)
    engineered = engineered.dropna()
    engineered = engineered.reset_index()
    return engineered


def prepare_train_test(
    features: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Chronological train/test split."""
    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    feature_cols = [c for c in features.columns if c not in {"Date", "Target"}]
    split_idx = max(int(len(features) * (1 - test_ratio)), 10)
    train = features.iloc[:split_idx].copy()
    test = features.iloc[split_idx:].copy()
    return train, test, feature_cols


def latest_feature_row(features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return most recent feature vector for inference."""
    if features.empty:
        return pd.DataFrame()
    row = features.iloc[[-1]][feature_cols]
    return row
