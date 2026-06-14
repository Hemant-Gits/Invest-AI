"""Shared UI helpers and styling."""

from __future__ import annotations

import streamlit as st


def inject_custom_css(theme: str = "light") -> None:
    bg = "#0e1117" if theme == "dark" else "#ffffff"
    text = "#fafafa" if theme == "dark" else "#1a1a1a"
    card = "#262730" if theme == "dark" else "#f8f9fa"
    accent = "#4f46e5"

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        .metric-card {{
            background: {card};
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border-left: 4px solid {accent};
            margin-bottom: 0.5rem;
        }}
        .status-normal {{ color: #2ecc71; font-weight: 600; }}
        .status-warning {{ color: #f39c12; font-weight: 600; }}
        .status-anomaly {{ color: #e74c3c; font-weight: 600; }}
        .research-box {{
            background: {card};
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid {accent};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = "") -> None:
    delta_html = f"<div style='font-size:0.85rem;opacity:0.8;'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.8rem;opacity:0.7;">{label}</div>
            <div style="font-size:1.4rem;font-weight:700;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> None:
    css_class = {
        "Normal": "status-normal",
        "Warning": "status-warning",
        "Anomaly": "status-anomaly",
    }.get(status, "status-normal")
    st.markdown(f'<span class="{css_class}">● {status}</span>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")
