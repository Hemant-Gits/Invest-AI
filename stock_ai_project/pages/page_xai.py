"""Page 4: Explainable AI & Model Analytics."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.explainability import explain_prediction, human_explanation
from models.sentiment_model import FinBERTSentimentAnalyzer
from utils.data_loader import fetch_news, fetch_stock_history
from utils.model_comparison import compare_models, compare_sentiment_research
from utils.ui_helpers import page_header


@st.cache_resource
def get_analyzer():
    return FinBERTSentimentAnalyzer()


def render():
    page_header(
        "Explainable AI & Model Analytics",
        "SHAP visualizations, model comparison, and research contribution analysis",
    )

    symbol = st.text_input("Stock Symbol", value="TCS.NS")
    period = st.select_slider("History", ["1y", "2y", "5y"], value="2y")

    if st.button("Run Model Analytics", type="primary"):
        with st.spinner("Training models and computing SHAP values..."):
            df = fetch_stock_history(symbol, period=period)
            if df.empty:
                st.error("Could not load stock data.")
                return

            analyzer = get_analyzer()
            news = fetch_news(symbol.replace(".NS", ""), 5)
            titles = [a.get("title", "") for a in news]
            sentiment_results = analyzer.analyze_batch(titles)
            agg = analyzer.aggregate_sentiment(sentiment_results)
            sentiment_score = agg.get("avg_score", 0.0)

            comparison = compare_models(df, sentiment_score=sentiment_score)
            if not comparison:
                st.error("Insufficient data for model comparison.")
                return

            st.session_state["xai_comparison"] = comparison
            st.session_state["xai_symbol"] = symbol
            st.session_state["xai_sentiment"] = sentiment_score

    comparison = st.session_state.get("xai_comparison")
    if not comparison:
        st.info("Enter a symbol and click **Run Model Analytics** to begin.")
        return

    st.subheader("Model Comparison Framework")
    metrics_df = pd.DataFrame(comparison["metrics"]).T
    metrics_df = metrics_df.round(4)
    st.dataframe(metrics_df, use_container_width=True)

    best = comparison["best_model"]
    st.success(f"Best Performing Model: **{best}** (lowest composite error score)")

    ranking = comparison.get("ranking", [])
    if ranking:
        rank_df = pd.DataFrame(ranking, columns=["Model", "Composite Score"])
        rank_df["Rank"] = range(1, len(rank_df) + 1)
        st.dataframe(rank_df, use_container_width=True)

    if comparison.get("test_dates") is not None and comparison.get("y_test") is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comparison["test_dates"], y=comparison["y_test"],
            name="Actual", line=dict(color="black"),
        ))
        for model_name, preds in comparison["predictions"].items():
            fig.add_trace(go.Scatter(
                x=comparison["test_dates"], y=preds,
                name=model_name, line=dict(dash="dot"),
            ))
        fig.update_layout(title="Model Predictions vs Actual", height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Research Contribution — Sentiment Impact Study")

    symbol = st.session_state.get("xai_symbol", "TCS.NS")
    df = fetch_stock_history(symbol, period="2y")
    sentiment_score = st.session_state.get("xai_sentiment", 0.0)

    if st.button("Run Sentiment Research Comparison"):
        with st.spinner("Comparing Model A (price only) vs Model B (price + sentiment)..."):
            research = compare_sentiment_research(df, sentiment_score)
            if research:
                st.session_state["research"] = research

    research = st.session_state.get("research")
    if research:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Model A — Price Features Only**")
            st.json(research["model_a"])
        with col_b:
            st.markdown("**Model B — Price + FinBERT Sentiment**")
            st.json(research["model_b"])

        improvement = research["improvement_pct"]
        if improvement > 0:
            st.success(f"Sentiment improved prediction accuracy by **{improvement:.2f}%** (RMSE reduction)")
        else:
            st.warning(f"Sentiment impact: {improvement:.2f}% (may vary by stock and news coverage)")

    st.markdown("---")
    st.subheader("SHAP Explainability Dashboard")

    xgb = comparison.get("xgb_model")
    train = comparison.get("train")
    test = comparison.get("test")
    feature_cols = comparison.get("feature_cols", [])

    if xgb and train is not None and test is not None and feature_cols:
        latest = test[feature_cols].iloc[[-1]]
        explanation = explain_prediction(xgb, train, latest, feature_cols)

        pred = float(xgb.predict(latest)[0])
        base = float(test["Target"].iloc[-2]) if len(test) > 1 else pred
        change_pct = ((pred - base) / base) * 100 if base else 0

        st.markdown("### Local Explanation")
        st.code(human_explanation(change_pct, explanation), language=None)

        exp_tab1, exp_tab2, exp_tab3 = st.tabs(["Summary Plot", "Feature Importance", "Waterfall"])
        with exp_tab1:
            st.pyplot(explanation["fig_summary"])
        with exp_tab2:
            st.pyplot(explanation["fig_importance"])
        with exp_tab3:
            st.pyplot(explanation["fig_waterfall"])

        st.markdown("### Feature Impact Table")
        st.dataframe(explanation["feature_impact"].round(4), use_container_width=True)
