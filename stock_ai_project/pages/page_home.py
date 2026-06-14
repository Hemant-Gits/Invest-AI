"""Page 1: Home / Education."""

import streamlit as st
from utils.data_loader import get_market_snapshot
from utils.ui_helpers import metric_card, page_header


def render():
    page_header(
        "Understanding Modern Markets",
        "AI-Powered Investment Decision Support — Final Year Engineering Project",
    )

    hero_left, hero_right = st.columns([1.5, 1])
    with hero_left:
        st.markdown("""
        Build a disciplined framework for allocating capital, balancing conviction with risk guardrails,
        and staying ahead of macro currents. **InvestAI** combines Machine Learning, Deep Learning,
        FinBERT Sentiment, SHAP Explainability, and RAG-based financial intelligence.
        """)
        st.markdown("**Core Pillars:** Wealth Acceleration • Financial Security • Inflation Shield")
        c1, c2 = st.columns(2)
        c1.info("Navigate via sidebar to explore all 6 modules")
        c2.success("Auto-refresh enabled every 60 seconds")

    with hero_right:
        st.image(
            "https://images.pexels.com/photos/6771607/pexels-photo-6771607.jpeg",
            caption="Global Capital Flows",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Live Market Pulse")
    snapshot = get_market_snapshot()
    cols = st.columns(3)
    for idx, (label, values) in enumerate(snapshot.items()):
        latest, pct = values
        if latest is None:
            cols[idx].metric(label, "N/A", "-")
        else:
            cols[idx].metric(label, f"{latest:,.2f}", f"{pct:+.2f}%")

    st.markdown("### Platform Modules")
    modules = st.columns(3)
    module_info = [
        ("Economic Analysis", "CPI trends, inflation impact on returns"),
        ("Stock Dashboard", "FinBERT sentiment, sector analytics, market mood"),
        ("Explainable AI", "SHAP visualizations, model comparison"),
        ("Stock Studio", "XGBoost + LSTM prediction engine, backtesting"),
        ("AI Analyst", "RAG chatbot with FAISS retrieval"),
        ("Portfolio Risk", "Multi-stock correlation and diversification"),
    ]
    for i, (title, desc) in enumerate(module_info):
        with modules[i % 3]:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("### Educational Resources")
    tabs = st.tabs(["Market Anatomy", "Why Invest", "Asset Mix", "Risk Radar"])
    with tabs[0]:
        st.markdown("""
        - **Equities** represent fractional ownership; prices track expected cash flows and risk appetite.
        - **Indices** like NIFTY or NASDAQ bundle leaders, offering diversified beta exposure.
        - **Market microstructure** blends long-term investors, quant funds, and retail traders.
        """)
    with tabs[1]:
        st.markdown("""
        - **Wealth Growth:** Reinvesting dividends plus appreciation compounds capital over time.
        - **Financial Security:** A funded portfolio cushions education, emergencies, and retirement.
        - **Inflation Hedge:** Equities historically beat CPI when companies have pricing power.
        """)
    with tabs[2]:
        st.markdown("""
        - **Large-cap:** Lower volatility, steady dividends, strong liquidity.
        - **Mid/small caps:** Higher growth potential with elevated risk.
        - **Alternatives:** Bonds, REITs, and gold stabilize drawdowns.
        """)
    with tabs[3]:
        st.markdown("""
        **Mitigation Playbook:** Position sizing, stop-losses, diversification, and trading journals.
        """)

    st.markdown("### Pre-Investment Checklist")
    c1, c2 = st.columns(2)
    with c1:
        st.checkbox("Defined investment goal", value=True)
        st.checkbox("Emergency fund in place (6–9 months)", value=True)
        st.checkbox("Diversification across sectors", value=True)
    with c2:
        st.checkbox("Risk appetite mapped to drawdown tolerance", value=True)
        st.checkbox("Systematic investment plan documented", value=True)
        st.checkbox("Exit strategy / rebalancing rules set", value=True)

    st.info("Tip: Use **Stock Studio** for AI predictions and **AI Analyst** for RAG-powered research queries.")
