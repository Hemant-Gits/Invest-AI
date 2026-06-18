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


def inject_studio_inputs_css() -> None:
    """Style Stock Studio analysis inputs like the dark reference panel."""
    primary = "#ff4b4b"
    panel_bg = "#0e1117"
    input_bg = "#262730"
    border = "#4a4d57"
    text = "#fafafa"
    panel = '[data-testid="stVerticalBlockBorderWrapper"]'

    st.markdown(
        f"""
        <style>
        /* Dark analysis inputs card */
        {panel} {{
            background-color: {panel_bg} !important;
            border: 1px solid #31333b !important;
            border-radius: 12px !important;
            padding: 1.1rem 1rem 1.25rem !important;
        }}
        {panel} h3, {panel} [data-testid="stHeader"] {{
            color: {text} !important;
        }}
        {panel} label[data-testid="stWidgetLabel"] p {{
            font-size: 0.9rem !important;
            color: {text} !important;
        }}
        {panel} [data-testid="stCaptionContainer"] p {{
            color: #a3a8b8 !important;
            font-size: 0.8rem !important;
        }}

        /* Quick-pick preset buttons */
        {panel} [data-testid="stHorizontalBlock"] button {{
            background: {input_bg} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            color: {text} !important;
            min-height: 2.4rem !important;
            white-space: nowrap !important;
        }}
        {panel} [data-testid="stHorizontalBlock"] button:hover {{
            border-color: {primary} !important;
        }}

        /* Quick-pick pills (legacy) */
        {panel} [data-testid="stPills"] {{
            gap: 0.45rem !important;
        }}
        {panel} [data-testid="stPills"] button {{
            background: {input_bg} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            color: {text} !important;
            padding: 0.45rem 0.65rem !important;
            min-height: 2.4rem !important;
            white-space: nowrap !important;
        }}
        {panel} [data-testid="stPills"] button[aria-pressed="true"] {{
            background: {primary} !important;
            border-color: {primary} !important;
            color: #ffffff !important;
        }}
        {panel} [data-testid="stPills"] button:hover {{
            border-color: {primary} !important;
        }}

        /* Text input + multiselect */
        {panel} [data-testid="stTextInput"] input,
        {panel} [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
            background: {input_bg} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            color: {text} !important;
        }}
        {panel} [data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
            background: #31333b !important;
            color: {text} !important;
        }}

        /* Lookback slider */
        {panel} [data-testid="stSlider"] [data-testid="stThumbValue"] {{
            color: {primary} !important;
        }}
        {panel} [data-testid="stSlider"] [role="slider"] {{
            background: {primary} !important;
        }}
        {panel} [data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
            background: {primary} !important;
        }}

        /* Action buttons */
        {panel} button[kind="primary"] {{
            background: {primary} !important;
            border: 1px solid {primary} !important;
            border-radius: 8px !important;
            color: #ffffff !important;
        }}
        {panel} button[kind="primary"]:hover {{
            background: #e04343 !important;
            border-color: #e04343 !important;
        }}
        {panel} [data-testid="stBaseButton-secondary"] {{
            background: transparent !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            color: {text} !important;
        }}
        {panel} [data-testid="stBaseButton-secondary"]:hover {{
            border-color: {primary} !important;
            color: {text} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")
