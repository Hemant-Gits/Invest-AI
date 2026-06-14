"""Market and economic data loading utilities."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from utils.config import CPI_DATA_PATH, MARKET_TICKERS, NEWS_API_KEY


def normalize_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to flat OHLCV columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.droplevel(1)
    out = out.reset_index()
    date_col = out.columns[0]
    if str(date_col).lower() != "date":
        out = out.rename(columns={date_col: "Date"})
    return out


def fetch_stock_history(
    symbol: str,
    period: str = "2y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download OHLCV history for a ticker."""
    try:
        if start and end:
            raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        else:
            raw = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        df = normalize_yfinance_df(raw)
        if df.empty:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_stock_info(symbol: str) -> dict:
    """Fetch ticker metadata from yfinance."""
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


def get_market_snapshot() -> dict[str, tuple[Optional[float], Optional[float]]]:
    """Return latest price and daily change for major indices."""
    snapshot = {}
    for label, ticker in MARKET_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if hist.empty:
                raise ValueError("empty")
            latest = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else latest
            pct = ((latest - prev) / prev) * 100 if prev else 0.0
            snapshot[label] = (latest, pct)
        except Exception:
            snapshot[label] = (None, None)
    return snapshot


def _normalize_cpi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean CPI column names and coerce numeric values."""
    out = df.copy()
    out.columns = [
        str(c).replace("\ufeff", "").strip().replace("  ", " ")
        for c in out.columns
    ]
    for col in out.columns:
        if col not in {"Year", "Month"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _cpi_candidate_paths() -> list[Path]:
    """Possible CPI CSV locations (project data dir, then repo root)."""
    project_root = CPI_DATA_PATH.parent.parent
    repo_root = project_root.parent
    names = ["All India Consumer Price Index.csv", "cpi_data.csv"]
    paths: list[Path] = []
    for base in (CPI_DATA_PATH.parent, project_root, repo_root):
        for name in names:
            candidate = base / name
            if candidate not in paths:
                paths.append(candidate)
    return paths


def load_cpi_data() -> pd.DataFrame:
    """Load India CPI dataset from the first available path."""
    for path in _cpi_candidate_paths():
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            df = _normalize_cpi_columns(df.dropna(how="all"))
            if df.empty:
                continue
            if "Year" not in df.columns or "Month" not in df.columns:
                continue
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception:
            continue
    return pd.DataFrame()


def fetch_news(stock_symbol: str, num_articles: int = 10) -> list[dict]:
    """Fetch news articles from NewsAPI."""
    if not NEWS_API_KEY:
        return []
    url = (
        f"https://newsapi.org/v2/everything?q={stock_symbol}"
        f"&apiKey={NEWS_API_KEY}&pageSize={num_articles}&language=en&sortBy=publishedAt"
    )
    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        return data.get("articles", [])
    except Exception:
        return []


def build_date_range(days: int) -> list[datetime]:
    """Build future business-day-like date range."""
    start = datetime.now() + timedelta(days=1)
    return pd.bdate_range(start=start, periods=days).tolist()
