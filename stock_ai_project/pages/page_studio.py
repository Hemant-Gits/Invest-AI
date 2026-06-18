"""Page 5: Stock Studio — Main AI Prediction Engine."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.anomaly_detection import AnomalyDetector
from models.explainability import explain_prediction, human_explanation, plot_local_importance
from models.sentiment_model import FinBERTSentimentAnalyzer
from models.xgboost_model import XGBoostStockModel
from utils.backtesting import run_backtest_report
from utils.config import PRESET_SYMBOLS
from utils.data_loader import fetch_news, fetch_stock_history
from utils.feature_engineering import engineer_features, latest_feature_row, prepare_train_test
from utils.model_comparison import compare_models
from utils.pdf_report import generate_stock_report
from utils.ui_helpers import inject_studio_inputs_css, page_header, status_badge
from utils.volatility_regime import analyze_volatility_regime


@st.cache_resource
def get_analyzer():
    return FinBERTSentimentAnalyzer()


PRESET_DISPLAY = {
    "TCS.NS": "TCS",
    "RELIANCE.NS": "Reliance",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC",
    "ICICIBANK.NS": "ICICI",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
}


def _preset_label(symbol: str) -> str:
    return PRESET_DISPLAY.get(symbol, symbol.split(".")[0])


def _symbol_caption(symbol: str) -> str:
    name = PRESET_DISPLAY.get(symbol)
    if name and name != symbol:
        return f"{name} · {symbol}"
    return symbol


def _render_preset_grid(presets: list[str]) -> None:
    """Quick-pick buttons in two rows — works across all supported Streamlit versions."""
    st.caption("Quick pick")
    row1_cols = st.columns(3)
    for i, preset in enumerate(presets[:3]):
        if row1_cols[i].button(
            _preset_label(preset),
            key=f"studio_preset_{preset}",
            use_container_width=True,
        ):
            st.session_state.symbol_input = preset

    row2_cols = st.columns(3)
    for i, preset in enumerate(presets[3:6]):
        if row2_cols[i].button(
            _preset_label(preset),
            key=f"studio_preset_{preset}_r2",
            use_container_width=True,
        ):
            st.session_state.symbol_input = preset


def _multi_stock_chart(symbols: list[str], period: str):
    fig = go.Figure()
    for sym in symbols:
        df = fetch_stock_history(sym, period=period)
        if df.empty:
            continue
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if close.empty:
            continue
        normalized = close / close.iloc[0] * 100
        dates = df.loc[close.index, "Date"] if "Date" in df.columns else close.index
        fig.add_trace(go.Scatter(x=dates, y=normalized, name=sym, mode="lines"))
    fig.update_layout(title="Multi-Stock Comparison (Normalized to 100)", height=450)
    return fig


def _sanitize_results_for_session(results: dict) -> dict:
    """Keep only Streamlit-safe data in session state (no matplotlib figures or model objects)."""
    explanation = results.get("explanation") or {}
    comparison = results.get("comparison") or {}
    sentiment = results.get("sentiment") or {}

    safe = dict(results)
    safe["explanation"] = {
        "positive_reasons": explanation.get("positive_reasons", []),
        "negative_reasons": explanation.get("negative_reasons", []),
        "feature_impact": explanation.get("feature_impact"),
    }
    safe["comparison"] = {"metrics": comparison.get("metrics", {})}
    safe["sentiment"] = {k: v for k, v in sentiment.items() if k != "results"}
    return safe


def _run_studio_analysis(symbol: str, period: str) -> dict | None:
    """Run the full Stock Studio pipeline and return all display artifacts."""
    try:
        df = fetch_stock_history(symbol, period=period)
        if df.empty:
            st.error("Failed to load stock data.")
            return None

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if close.empty:
            st.error("No valid closing prices found for this symbol.")
            return None
        base_price = float(close.iloc[-1])

        analyzer = get_analyzer()
        news = fetch_news(symbol.replace(".NS", "").replace(".BO", ""), 8)
        titles = [a.get("title", "") for a in news]
        sentiment_agg = analyzer.aggregate_sentiment(analyzer.analyze_batch(titles))
        sentiment_score = sentiment_agg.get("avg_score", 0.0)

        features = engineer_features(df, sentiment_score=sentiment_score)
        train, test, feature_cols = prepare_train_test(features)
        if train.empty:
            st.error("Insufficient data for feature engineering.")
            return None

        xgb = XGBoostStockModel()
        xgb.fit(train[feature_cols], train["Target"])
        latest = latest_feature_row(features, feature_cols)
        horizons = xgb.predict_horizons(latest, feature_cols, base_price)

        comparison = compare_models(df, sentiment_score=sentiment_score)
        prices = pd.to_numeric(
            df.set_index("Date")["Close"] if "Date" in df.columns else df["Close"],
            errors="coerce",
        ).dropna()
        vol = analyze_volatility_regime(prices)
        detector = AnomalyDetector()
        anomaly_df = detector.fit_predict(df)
        anomaly_status = detector.latest_status(anomaly_df)

        y_pred = xgb.predict(test[feature_cols])
        backtest = run_backtest_report(test["Target"], pd.Series(y_pred, index=test.index))

        explanation = explain_prediction(xgb, train, latest, feature_cols)
        change_pct = horizons.get(1, {}).get("change_pct", 0)

        return _sanitize_results_for_session({
            "symbol": symbol,
            "period": period,
            "base_price": base_price,
            "horizons": horizons,
            "sentiment": sentiment_agg,
            "vol": vol,
            "anomaly_status": anomaly_status,
            "anomaly_df": anomaly_df,
            "anomaly_count": detector.count_anomalies(anomaly_df),
            "backtest": backtest,
            "comparison": comparison,
            "explanation": explanation,
            "change_pct": change_pct,
        })
    except Exception as exc:
        st.error(f"Stock Studio analysis failed: {exc}")
        return None


def _render_results(results: dict, gen_pdf: bool) -> None:
    symbol = results["symbol"]
    currency = "₹" if ".NS" in symbol.upper() or ".BO" in symbol.upper() else "$"
    base_price = results["base_price"]
    horizons = results["horizons"]
    sentiment_agg = results["sentiment"]
    vol = results["vol"]
    anomaly_status = results["anomaly_status"]
    anomaly_df = results["anomaly_df"]
    backtest = results["backtest"]
    comparison = results["comparison"]
    explanation = results["explanation"]
    change_pct = results["change_pct"]

    pulse_tab, forecast_tab, risk_tab, backtest_tab, xai_tab = st.tabs([
        "Market Pulse", "Forecasts", "Risk Analytics", "Backtesting", "Explainability"
    ])

    with pulse_tab:
        c1, c2 = st.columns(2)
        c1.metric("Spot Price", f"{currency}{base_price:,.2f}")
        c2.metric("Market Mood", sentiment_agg.get("market_mood", "N/A"))
        c3, c4 = st.columns(2)
        c3.metric("Volatility", vol.get("Regime", "N/A"))
        with c4:
            st.markdown("**Anomaly**")
            status_badge(anomaly_status)

    with forecast_tab:
        st.subheader("XGBoost Multi-Horizon Predictions")
        for horizon in [1, 7, 30]:
            h = horizons.get(horizon, {})
            st.metric(
                f"{horizon}-Day Prediction",
                f"{currency}{h.get('predicted_price', 0):,.2f}",
                f"{h.get('change_pct', 0):+.2f}%",
            )
            st.caption(
                f"95% Confidence Interval: [{currency}{h.get('lower', 0):,.2f} – "
                f"{currency}{h.get('upper', 0):,.2f}]"
            )

        path = horizons.get("path", [])
        if path:
            path_df = pd.DataFrame(path)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=path_df["day"], y=path_df["price"], name="Forecast", line=dict(color="#4f46e5")))
            fig.add_trace(go.Scatter(
                x=path_df["day"], y=path_df["upper"],
                fill=None, mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=path_df["day"], y=path_df["lower"],
                fill="tonexty", mode="lines", line=dict(width=0),
                name="95% CI", fillcolor="rgba(79,70,229,0.2)",
            ))
            fig.update_layout(title="Prediction with Confidence Bands", xaxis_title="Days Ahead")
            st.plotly_chart(fig, use_container_width=True)

    with risk_tab:
        st.markdown(f"**Volatility Regime:** {vol.get('Regime')} — {vol.get('Explanation')}")
        vol_series = vol.get("Volatility Series")
        if vol_series is not None and not vol_series.empty:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(y=vol_series.values, mode="lines", name="Rolling Volatility"))
            fig_vol.update_layout(title="Rolling Volatility (Annualized)")
            st.plotly_chart(fig_vol, use_container_width=True)

        if not anomaly_df.empty:
            date_col = "Date" if "Date" in anomaly_df.columns else anomaly_df.columns[0]
            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(
                x=anomaly_df[date_col], y=anomaly_df["Return"],
                mode="markers",
                marker=dict(
                    color=anomaly_df["Status"].map({"Normal": "green", "Warning": "orange", "Anomaly": "red"}),
                    size=8,
                ),
                name="Daily Returns",
            ))
            fig_anom.update_layout(title="Anomaly Detection on Returns")
            st.plotly_chart(fig_anom, use_container_width=True)
            st.caption(f"Anomalies detected: {results.get('anomaly_count', 0)}")

    with backtest_tab:
        if backtest:
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Sharpe Ratio", f"{backtest.get('Sharpe Ratio', 0):.2f}")
            bc2.metric("Max Drawdown", f"{backtest.get('Max Drawdown', 0):.2f}%")
            bc3.metric("Risk Level", backtest.get("Risk Recommendation", "N/A"))

            metrics_show = ["Sortino Ratio", "Win Rate", "CAGR", "Total Return"]
            for m in metrics_show:
                st.write(f"**{m}:** {backtest.get(m, 0):.2f}{'%' if m != 'Sortino Ratio' and m != 'Sharpe Ratio' else ''}")

            equity = backtest.get("Equity_Curve")
            if equity is not None and not equity.empty:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=equity["Equity_Curve"], name="Equity Curve", fill="tozeroy"))
                rolling_max = equity["Equity_Curve"].cummax()
                drawdown = (equity["Equity_Curve"] - rolling_max) / rolling_max * 100
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(y=drawdown, name="Drawdown %", fill="tozeroy", line=dict(color="red")))
                st.plotly_chart(fig_eq, use_container_width=True)
                st.plotly_chart(fig_dd, use_container_width=True)

        if comparison:
            st.subheader("Model Comparison")
            st.dataframe(pd.DataFrame(comparison["metrics"]).T.round(4), use_container_width=True)

    with xai_tab:
        st.code(human_explanation(change_pct, explanation), language=None)
        feature_impact = explanation.get("feature_impact")
        if feature_impact is not None and not feature_impact.empty:
            st.pyplot(plot_local_importance(feature_impact))

    if gen_pdf:
        pdf_path = generate_stock_report(
            stock_name=symbol,
            prediction=horizons,
            sentiment=sentiment_agg,
            anomaly_status=anomaly_status,
            volatility=vol,
            backtest=backtest,
            model_metrics=comparison.get("metrics", {}),
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF Report",
                data=f.read(),
                file_name=pdf_path.name,
                mime="application/pdf",
            )
        st.success(f"Report generated: {pdf_path.name}")


def render():
    page_header(
        "Stock Intelligence Studio",
        "XGBoost + LSTM prediction engine with backtesting, anomaly detection, and PDF reports",
    )

    inject_studio_inputs_css()

    if "symbol_input" not in st.session_state:
        st.session_state.symbol_input = "TCS.NS"

    config_col, main_col = st.columns([1.45, 2.05])

    with config_col:
        with st.container(border=True):
            st.subheader("Analysis Inputs")

            _render_preset_grid(PRESET_SYMBOLS[:6])

            symbol = st.text_input(
                "Stock Symbol",
                key="symbol_input",
                help="NSE stocks use .NS suffix (e.g. TCS.NS). US stocks use plain tickers (e.g. AAPL).",
            )
            st.caption(_symbol_caption(symbol))
            period = st.select_slider("Lookback", ["6mo", "1y", "2y", "5y"], value="2y")
            compare_symbols = st.multiselect(
                "Compare Stocks",
                PRESET_SYMBOLS,
                default=[],
                placeholder="Choose options",
                format_func=lambda s: f"{PRESET_DISPLAY.get(s, s.split('.')[0])} ({s})",
            )

            analyze = st.button("Run Full Analysis", type="primary", use_container_width=True)
            gen_pdf = st.button("Generate PDF Report", use_container_width=True)

    with main_col:
        if compare_symbols:
            st.plotly_chart(_multi_stock_chart(compare_symbols, period), use_container_width=True)

        if analyze or gen_pdf:
            with st.spinner("Running XGBoost, sentiment analysis, anomaly detection... (first run may take up to a minute)"):
                results = _run_studio_analysis(symbol, period)
                if results:
                    st.session_state["studio_results"] = results
                else:
                    st.session_state.pop("studio_results", None)
                    return
        elif not st.session_state.get("studio_results"):
            st.info("Configure inputs and click **Run Full Analysis** to launch the AI prediction engine.")
            return

        results = st.session_state.get("studio_results")
        if results:
            _render_results(results, gen_pdf=gen_pdf)
