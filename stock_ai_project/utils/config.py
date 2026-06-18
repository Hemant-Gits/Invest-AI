"""Application configuration and constants."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"

CPI_DATA_PATH = DATA_DIR / "All India Consumer Price Index.csv"

def _get_secret(key: str, default: str = "") -> str:
    """Read config from Streamlit secrets, then environment variables."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


NEWS_API_KEY = _get_secret("NEWS_API_KEY", "")
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY", "")


def get_news_api_key() -> str:
    """Read latest NewsAPI key from secrets/env at runtime."""
    return _get_secret("NEWS_API_KEY", "")


def is_news_api_key_configured() -> bool:
    """Return True only when a real NewsAPI key is configured."""
    key = (get_news_api_key() or NEWS_API_KEY or "").strip()
    if not key:
        return False
    placeholders = {
        "your_newsapi_key_here",
        "your_key_from_newsapi.org",
        "your_key",
        "changeme",
        "replace_me",
    }
    lowered = key.lower()
    return lowered not in placeholders and not lowered.startswith("your_")

NIFTY100_SECTORS = {
    "Technology": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS", "WIPRO.NS"],
    "Oil & Energy": ["RELIANCE.NS", "ONGC.NS", "IOC.NS"],
    "Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "SBI.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "SBI.NS", "KOTAKBANK.NS"],
    "Automotive": ["TATAMOTORS.NS", "MARUTI.NS", "HEROMOTOCO.NS"],
}

PRESET_SYMBOLS = [
    "TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "AAPL", "MSFT"
]

MARKET_TICKERS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NASDAQ 100": "^NDX",
}

NEWS_STOCKS = [
    "TCS", "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "BAJFINANCE", "KOTAKBANK", "ITC", "ASIANPAINT",
]

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "Lag_1", "Lag_3", "Lag_5", "Lag_10",
    "Rolling_Mean_5", "Rolling_Mean_10", "Rolling_Mean_20",
    "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "Volatility",
]
