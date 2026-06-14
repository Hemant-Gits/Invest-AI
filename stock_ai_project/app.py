"""
InvestAI — AI-Powered Stock Market Prediction, Analysis and Decision Support System
Final Year Engineering Project
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from pages import page_chatbot, page_dashboard, page_home, page_inflation, page_studio, page_xai
from utils.ui_helpers import inject_custom_css

PAGES = {
    "Home / Education": page_home,
    "Economic Analysis": page_inflation,
    "Stock Dashboard": page_dashboard,
    "Explainable AI": page_xai,
    "Stock Studio": page_studio,
    "AI Stock Analyst": page_chatbot,
}

PAGE_ICONS = {
    "Home / Education": "🏠",
    "Economic Analysis": "📊",
    "Stock Dashboard": "📈",
    "Explainable AI": "🔬",
    "Stock Studio": "🎯",
    "AI Stock Analyst": "🤖",
}


def init_session():
    defaults = {
        "theme": "light",
        "auto_refresh": True,
        "last_refresh": time.time(),
        "selected_page": "Home / Education",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def sidebar_navigation():
    with st.sidebar:
        st.markdown("## Invest AI")
        st.caption("AI Stock Market Decision Support System")
        st.markdown("---")

        theme = st.toggle("Dark Mode", value=st.session_state.theme == "dark")
        st.session_state.theme = "dark" if theme else "light"

        auto_refresh = st.toggle("Auto Refresh (60s)", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh

        st.markdown("---")
        st.markdown("### Navigation")

        for name in PAGES:
            icon = PAGE_ICONS.get(name, "•")
            if st.button(f"{icon} {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state.selected_page = name

        st.markdown("---")
        st.markdown("### Tech Stack")
        st.markdown("""
        - XGBoost & LSTM
        - FinBERT NLP
        - SHAP XAI
        - FAISS + LangChain RAG
        - Plotly Visualizations
        """)

        st.markdown("---")
        st.caption("Final Year Engineering Project")
        st.caption("For academic use only — not financial advice.")


def main():
    st.set_page_config(
        page_title="Invest AI — Stock Market Decision Support",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session()
    inject_custom_css(st.session_state.theme)
    sidebar_navigation()

    selected = st.session_state.selected_page
    st.markdown(f"# {PAGE_ICONS.get(selected, '')} {selected}")

    page_module = PAGES.get(selected)
    if page_module:
        page_module.render()

    if st.session_state.auto_refresh:
        elapsed = time.time() - st.session_state.last_refresh
        if elapsed >= 60:
            st.session_state.last_refresh = time.time()
            st.rerun()


if __name__ == "__main__":
    main()
