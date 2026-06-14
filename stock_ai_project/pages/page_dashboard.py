"""Page 3: Stock Dashboard with FinBERT sentiment."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from models.sentiment_model import FinBERTSentimentAnalyzer
from utils.config import NEWS_STOCKS, NIFTY100_SECTORS
from utils.data_loader import fetch_news, fetch_stock_history, fetch_stock_info
from utils.ui_helpers import page_header, status_badge


@st.cache_resource
def get_sentiment_analyzer():
    return FinBERTSentimentAnalyzer()


def _load_sector_data(sector: str) -> pd.DataFrame:
    stocks = NIFTY100_SECTORS.get(sector, [])
    rows = []
    for stock in stocks:
        df = fetch_stock_history(stock, period="5d")
        if df.empty:
            continue
        info = fetch_stock_info(stock)
        rows.append({
            "Stock": stock,
            "Current Price": df["Close"].iloc[-1],
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
        })
    return pd.DataFrame(rows)


def render():
    page_header(
        "Stock Dashboard — FinBERT Sentiment Intelligence",
        "Sector analytics, finance-specific NLP, and market mood indicators",
    )

    tab_overview, tab_news, tab_safety, tab_portfolio = st.tabs([
        "Sector Overview", "FinBERT News Sentiment", "Stock Safety", "Portfolio Risk"
    ])

    with tab_overview:
        sector = st.selectbox("Select Sector", list(NIFTY100_SECTORS.keys()))
        sector_data = _load_sector_data(sector)

        if sector_data.empty:
            st.warning("No data available for this sector.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(sector_data, use_container_width=True)
                top = sector_data.nlargest(5, "Current Price")
                st.subheader("Top 5 by Price")
                st.dataframe(top)
            with c2:
                fig = px.bar(
                    sector_data.dropna(subset=["Market Cap"]),
                    x="Stock", y="Market Cap",
                    title=f"{sector} — Market Cap",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab_news:
        st.subheader("FinBERT Financial Sentiment Engine")
        analyzer = get_sentiment_analyzer()
        selected = st.selectbox("Select Stock", NEWS_STOCKS)
        num_articles = st.slider("Articles to analyze", 5, 20, 10)

        if st.button("Analyze Sentiment", type="primary"):
            with st.spinner("Fetching news and running FinBERT analysis..."):
                articles = fetch_news(selected, num_articles)
                if not articles:
                    st.warning("No news found. Check NEWS_API_KEY or try another symbol.")
                else:
                    titles = [a.get("title", "") for a in articles]
                    results = analyzer.analyze_batch(titles)
                    agg = analyzer.aggregate_sentiment(results)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Market Mood", agg["market_mood"])
                    m2.metric("Positive %", f"{agg['positive_pct']:.1f}%")
                    m3.metric("Negative %", f"{agg['negative_pct']:.1f}%")
                    m4.metric("Avg Sentiment", f"{agg['avg_score']:.3f}")

                    mood_color = {"Bullish": "green", "Bearish": "red", "Neutral": "orange"}
                    st.markdown(
                        f"**Market Mood Indicator:** :{mood_color.get(agg['market_mood'], 'orange')}[{agg['market_mood']}]"
                    )

                    sentiment_rows = []
                    for article, result in zip(articles, results):
                        sentiment_rows.append({
                            "Title": article.get("title", "")[:80],
                            "Sentiment": result.label.capitalize(),
                            "Score": f"{result.polarity:.3f}",
                            "Confidence": f"{result.confidence:.1%}",
                            "URL": article.get("url", ""),
                        })

                    st.dataframe(pd.DataFrame(sentiment_rows), use_container_width=True)

                    pie_data = pd.DataFrame({
                        "Sentiment": ["Positive", "Negative", "Neutral"],
                        "Percentage": [agg["positive_pct"], agg["negative_pct"], agg["neutral_pct"]],
                    })
                    fig_pie = px.pie(pie_data, values="Percentage", names="Sentiment", title="Sentiment Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True)

                    trend_data = pd.DataFrame(sentiment_rows)
                    trend_data["Index"] = range(len(trend_data))
                    fig_trend = px.bar(
                        trend_data, x="Index", y="Score", color="Sentiment",
                        title="Sentiment Trend Across Recent Articles",
                        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

                    st.session_state["last_sentiment"] = agg

    with tab_safety:
        st.subheader("Stock Safety Calculator (P/E Based)")
        ticker = st.text_input("Enter ticker (e.g., TCS)", value="TCS")
        if ticker:
            sym = ticker.upper() + (".NS" if not ticker.endswith(".NS") else "")
            info = fetch_stock_info(sym)
            pe = info.get("trailingPE", 0)
            price = info.get("regularMarketPrice", info.get("currentPrice", 0))

            if pe and pe > 0:
                safe = pe <= 20
                st.metric("P/E Ratio", f"{pe:.2f}")
                st.metric("Current Price", f"₹{price:.2f}" if price else "N/A")
                if safe:
                    st.success("Stock appears reasonably valued (P/E ≤ 20)")
                else:
                    st.warning(f"P/E above 20 — further fundamental analysis recommended")
            else:
                st.warning("P/E ratio unavailable for this ticker.")

    with tab_portfolio:
        st.subheader("Portfolio Risk Dashboard")
        st.caption("Select 3–5 stocks for portfolio analytics")
        default = ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS"]
        symbols = st.multiselect("Portfolio Stocks", list(set(s for stocks in NIFTY100_SECTORS.values() for s in stocks)), default=default)

        if len(symbols) < 2:
            st.info("Select at least 2 stocks for portfolio analysis.")
        elif st.button("Analyze Portfolio"):
            from utils.portfolio_risk import portfolio_metrics

            with st.spinner("Computing portfolio metrics..."):
                metrics = portfolio_metrics(symbols)
                if not metrics:
                    st.error("Could not compute portfolio metrics.")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Portfolio Return (ann.)", f"{metrics['Portfolio Return']:.2f}%")
                    c2.metric("Portfolio Volatility", f"{metrics['Portfolio Volatility']:.2f}%")
                    c3.metric("Diversification Score", f"{metrics['Diversification Score']:.1f}/100")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        weights = metrics["Weights"]
                        fig_pie = px.pie(
                            values=list(weights.values()), names=list(weights.keys()),
                            title="Portfolio Allocation",
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_b:
                        corr = metrics["Correlation Matrix"]
                        fig_heat = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap", color_continuous_scale="RdBu_r")
                        st.plotly_chart(fig_heat, use_container_width=True)

                    returns = metrics["Returns"]
                    risk_return = pd.DataFrame({
                        "Stock": returns.columns,
                        "Return": returns.mean() * 252 * 100,
                        "Volatility": returns.std() * (252 ** 0.5) * 100,
                    })
                    fig_scatter = px.scatter(
                        risk_return, x="Volatility", y="Return", text="Stock",
                        title="Risk-Return Scatter Plot", size=[20] * len(risk_return),
                    )
                    fig_scatter.update_traces(textposition="top center")
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    st.session_state["portfolio_metrics"] = metrics
