"""Page 6: AI Stock Analyst — RAG Chatbot."""

import streamlit as st

from chatbot.rag_engine import RAGEngine
from utils.config import PRESET_SYMBOLS
from utils.ui_helpers import page_header


@st.cache_resource
def get_rag_engine():
    return RAGEngine()


def render():
    page_header(
        "AI Stock Analyst — RAG-Powered Chatbot",
        "Retrieval-Augmented Generation using FAISS, LangChain, FinBERT, and XGBoost intelligence",
    )

    engine = get_rag_engine()

    st.markdown("""
    Ask investment research questions. The chatbot retrieves:
    - Latest news and **FinBERT** sentiment scores
    - **XGBoost** price predictions with confidence intervals
    - Backtesting metrics and risk indicators
    - Volatility regime and anomaly status
    """)

    symbol = st.selectbox("Primary Stock Symbol", PRESET_SYMBOLS, index=0)
    example_queries = [
        f"Should I buy {symbol.replace('.NS', '')}?",
        f"What is the sentiment for {symbol.replace('.NS', '')}?",
        f"What are the risk metrics for {symbol}?",
        f"What does the AI model predict for {symbol}?",
    ]

    st.caption("Example queries:")
    eq_cols = st.columns(2)
    for i, eq in enumerate(example_queries):
        if eq_cols[i % 2].button(eq, key=f"eq_{i}"):
            st.session_state["chat_input"] = eq

    if st.button("Build Knowledge Index", type="secondary"):
        with st.spinner(f"Indexing intelligence for {symbol}..."):
            context = engine.build_context(symbol)
            if "error" not in context:
                st.success(f"FAISS index built with {len(context.get('documents', []))} documents.")
                st.session_state["rag_context"] = context
            else:
                st.error(context["error"])

    user_query = st.text_input(
        "Your Question",
        value=st.session_state.get("chat_input", ""),
        placeholder="Should I buy Reliance?",
        key="chat_input_box",
    )

    if st.button("Ask AI Analyst", type="primary") or user_query:
        if not user_query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving context and generating analysis..."):
            response = engine.query(user_query, symbol=symbol)

        st.markdown("### AI Analyst Response")
        st.markdown(response["answer"])

        if response.get("sources"):
            with st.expander("Retrieved Sources (RAG Context)"):
                for i, source in enumerate(response["sources"], 1):
                    st.markdown(f"{i}. {source}")

        ctx = response.get("context", {})
        if ctx:
            c1, c2, c3 = st.columns(3)
            sentiment = ctx.get("sentiment", {})
            c1.metric("Sentiment Mood", sentiment.get("market_mood", "N/A"))
            c2.metric("Anomaly Status", ctx.get("anomaly_status", "N/A"))
            c3.metric("Volatility", ctx.get("volatility", {}).get("Regime", "N/A"))

    st.markdown("---")
    st.markdown("""
    **RAG Architecture:**
    1. Data ingestion (news, prices, predictions, risk metrics)
    2. Document embedding via HuggingFace sentence-transformers
    3. FAISS vector indexing for semantic retrieval
    4. LangChain retrieval pipeline
    5. Context-aware structured response generation
    """)
