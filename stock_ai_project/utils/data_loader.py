"""Market and economic data loading utilities."""

from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import quote_plus
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import yfinance as yf

from utils.config import (
    CPI_DATA_PATH,
    MARKET_TICKERS,
    NEWS_API_KEY,
    get_news_api_key,
    is_news_api_key_configured,
)


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
        if "Close" in df.columns:
            df = df.dropna(subset=["Close"])
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


def _normalize_stock_symbol(stock_symbol: str) -> str:
    """Normalize user/dashboard symbols to bare tickers."""
    return (stock_symbol or "").strip().upper().replace(".NS", "").replace(".BO", "")


def _yfinance_symbol_candidates(stock_symbol: str) -> list[str]:
    """Try NSE, BSE, and plain symbols for Yahoo Finance news."""
    symbol = (stock_symbol or "").strip().upper()
    if not symbol:
        return []

    base = _normalize_stock_symbol(symbol)
    candidates: list[str] = []
    if "." in symbol:
        candidates.append(symbol)
    candidates.extend([f"{base}.NS", f"{base}.BO", base])

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _company_name(stock_symbol: str) -> str:
    symbol = _normalize_stock_symbol(stock_symbol)
    company_aliases = {
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "RELIANCE": "Reliance Industries",
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank",
        "KOTAKBANK": "Kotak Mahindra Bank",
        "BAJFINANCE": "Bajaj Finance",
        "HINDUNILVR": "Hindustan Unilever",
        "ASIANPAINT": "Asian Paints",
        "ITC": "ITC",
    }
    return company_aliases.get(symbol, symbol)


def _extract_yfinance_article(item: dict) -> Optional[dict]:
    """Normalize Yahoo Finance news payloads to NewsAPI-like article dicts."""
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = (content.get("title") or item.get("title") or "").strip()
    if not title:
        return None

    url = ""
    for url_key in ("clickThroughUrl", "canonicalUrl"):
        candidate = content.get(url_key)
        if isinstance(candidate, dict):
            url = candidate.get("url", "") or url
        elif isinstance(candidate, str):
            url = candidate

    provider = content.get("provider") or {}
    source_name = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"

    return {
        "title": title,
        "url": url,
        "source": {"name": source_name},
        "_provider": "yahoo_finance",
    }


def _fetch_news_yfinance(stock_symbol: str, num_articles: int) -> list[dict]:
    """Fetch finance news via yfinance (no API key required)."""
    articles: list[dict] = []
    for symbol in _yfinance_symbol_candidates(stock_symbol):
        try:
            raw_news = yf.Ticker(symbol).news or []
        except Exception:
            continue

        for item in raw_news:
            article = _extract_yfinance_article(item)
            if article and not any(a.get("title") == article.get("title") for a in articles):
                articles.append(article)
            if len(articles) >= num_articles:
                return articles
    return articles


def _fetch_news_rss(stock_symbol: str, num_articles: int) -> list[dict]:
    """Fetch finance headlines from Google News RSS (no API key required)."""
    company = _company_name(stock_symbol)
    query = quote_plus(f"{company} stock India")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; InvestAI/1.0)"}

    try:
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return []

    articles: list[dict] = []
    for item in root.findall(".//item"):
        title_node = item.find("title")
        link_node = item.find("link")
        title = (title_node.text or "").strip() if title_node is not None else ""
        link = (link_node.text or "").strip() if link_node is not None else ""
        if not title:
            continue
        articles.append({
            "title": title,
            "url": link,
            "source": {"name": "Google News"},
            "_provider": "google_news",
        })
        if len(articles) >= num_articles:
            break
    return articles


def _fetch_news_newsapi(stock_symbol: str, num_articles: int, api_key: str) -> list[dict]:
    """Fetch finance news articles from NewsAPI."""
    base_url = "https://newsapi.org/v2/everything"
    page_size = max(1, min(int(num_articles), 30))

    for query in _build_news_queries(stock_symbol):
        url = (
            f"{base_url}?q={quote_plus(query)}"
            f"&apiKey={api_key}&pageSize={page_size}&language=en&sortBy=publishedAt"
        )
        try:
            response = requests.get(url, timeout=15)
            data = response.json()
            if response.status_code != 200 or data.get("status") != "ok":
                continue

            articles = data.get("articles", [])
            if articles:
                for article in articles:
                    article["_provider"] = "newsapi"
                return articles
        except Exception:
            continue

    return []


def _build_news_queries(stock_symbol: str) -> list[str]:
    """Create increasingly broad finance-focused queries for NewsAPI."""
    symbol = _normalize_stock_symbol(stock_symbol)
    if not symbol:
        return []

    company = _company_name(stock_symbol)

    return [
        f"({company} OR {symbol}) AND (stock OR shares OR earnings OR results OR market)",
        f"{company} stock",
        symbol,
    ]


def fetch_news(stock_symbol: str, num_articles: int = 10) -> list[dict]:
    """Fetch finance news using Yahoo Finance, Google News RSS, then NewsAPI."""
    limit = max(1, min(int(num_articles), 30))

    articles = _fetch_news_yfinance(stock_symbol, limit)
    if articles:
        return articles

    articles = _fetch_news_rss(stock_symbol, limit)
    if articles:
        return articles

    if is_news_api_key_configured():
        api_key = get_news_api_key() or NEWS_API_KEY
        articles = _fetch_news_newsapi(stock_symbol, limit, api_key)
        if articles:
            return articles

    return []


def build_date_range(days: int) -> list[datetime]:
    """Build future business-day-like date range."""
    start = datetime.now() + timedelta(days=1)
    return pd.bdate_range(start=start, periods=days).tolist()
