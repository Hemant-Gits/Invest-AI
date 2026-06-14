"""Page 2: Economic Analysis / CPI & Inflation."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_cpi_data
from utils.ui_helpers import page_header


def _cpi_value_columns(df: pd.DataFrame) -> list[str]:
    """Return CPI numeric columns, excluding Year/Month/Date."""
    skip = {"Year", "Month", "Date"}
    return [
        col for col in df.columns
        if col not in skip and pd.api.types.is_numeric_dtype(df[col])
    ]


def _prepare_cpi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CPI data with a parsed Date column."""
    if df.empty:
        return pd.DataFrame()

    data = df.copy()
    data.columns = [str(c).replace("\ufeff", "").strip() for c in data.columns]

    if "Year" not in data.columns or "Month" not in data.columns:
        return pd.DataFrame()

    month_str = data["Month"].astype(str).str.strip()
    data["Date"] = pd.to_datetime(
        data["Year"].astype(str) + " " + month_str,
        format="%Y %b",
        errors="coerce",
    )
    if data["Date"].isna().all():
        data["Date"] = pd.to_datetime(
            data["Year"].astype(str) + "-" + month_str,
            errors="coerce",
        )

    data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return data


def _to_long_format(data: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Melt CPI data to long format — avoids fragile multi-column indexing."""
    valid = [c for c in value_cols if c in data.columns]
    if not valid or "Date" not in data.columns:
        return pd.DataFrame(columns=["Date", "Category", "CPI"])
    return data[["Date"] + valid].melt(
        id_vars="Date",
        value_vars=valid,
        var_name="Category",
        value_name="CPI",
    )


def _sanitize_selection(requested, allowed: list[str]) -> list[str]:
    """Keep only selections that exist in the current dataset."""
    allowed_set = set(allowed)
    return [item for item in requested if item in allowed_set]


def render():
    page_header(
        "Economic Analysis — Inflation & CPI Trends",
        "Analytics-focused economic dashboard (Prophet removed — trend analysis replaces forecasting)",
    )

    raw_data = load_cpi_data()
    if raw_data.empty:
        st.error(
            "CPI dataset not found. Place 'All India Consumer Price Index.csv' in "
            "`stock_ai_project/data/` or the InvestAI root folder."
        )
        return

    data = _prepare_cpi_data(raw_data)
    value_cols = _cpi_value_columns(data)

    if data.empty or not value_cols:
        st.error("Could not parse CPI data. Check that the CSV has Year, Month, and numeric CPI columns.")
        with st.expander("Debug: raw file columns"):
            st.write(list(raw_data.columns))
            st.dataframe(raw_data.head())
        return

    cpi_long = _to_long_format(data, value_cols)

    selected_category = st.selectbox(
        "Select CPI Category",
        options=value_cols,
        index=0,
        key="cpi_selected_category",
    )
    selected_category = _sanitize_selection([selected_category], value_cols)
    if not selected_category:
        st.error("No valid CPI category selected.")
        return
    selected_category = selected_category[0]

    chart_col, calc_col = st.columns([2, 1])

    with chart_col:
        series = cpi_long[cpi_long["Category"] == selected_category].copy()
        series = series.sort_values("Date")
        series["YoY_Change"] = series["CPI"].pct_change(periods=12) * 100
        series["Rolling_Avg"] = series["CPI"].rolling(6, min_periods=1).mean()

        fig = px.line(
            series,
            x="Date",
            y="CPI",
            title=f"CPI Trend — {selected_category}",
        )
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["YoY Inflation Rate", "Rolling Average", "Category Comparison"])
        with tab1:
            yoy_data = series.dropna(subset=["YoY_Change"])
            if yoy_data.empty:
                st.info("Not enough history to compute year-over-year inflation.")
            else:
                fig_yoy = px.bar(
                    yoy_data,
                    x="Date",
                    y="YoY_Change",
                    title="Year-over-Year Inflation Rate (%)",
                    color="YoY_Change",
                    color_continuous_scale=["green", "yellow", "red"],
                )
                st.plotly_chart(fig_yoy, use_container_width=True)

        with tab2:
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(x=series["Date"], y=series["CPI"], name="CPI"))
            fig_roll.add_trace(go.Scatter(x=series["Date"], y=series["Rolling_Avg"], name="6M Avg"))
            fig_roll.update_layout(title="CPI with 6-Month Rolling Average")
            st.plotly_chart(fig_roll, use_container_width=True)

        with tab3:
            compare_cols = st.multiselect(
                "Compare categories",
                options=value_cols,
                default=value_cols[: min(3, len(value_cols))],
                key="cpi_compare_categories",
            )
            compare_cols = _sanitize_selection(compare_cols, value_cols)

            if not compare_cols:
                st.info("Select at least one category to compare.")
            else:
                compare_df = cpi_long[cpi_long["Category"].isin(compare_cols)].copy()
                if compare_df.empty:
                    st.warning("No data available for the selected categories.")
                else:
                    fig_cmp = px.line(
                        compare_df,
                        x="Date",
                        y="CPI",
                        color="Category",
                        title="Multi-Category CPI Comparison",
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

    with calc_col:
        st.subheader("Inflation Impact Calculator")
        st.caption("Estimate how inflation erodes real investment returns")

        investment = st.number_input("Investment Amount (INR)", min_value=1000, value=100000, step=1000)
        years = st.slider("Investment Horizon (years)", 1, 30, 10)
        expected_return = st.slider("Expected Nominal Return (%/year)", 1.0, 20.0, 12.0)
        inflation_rate = st.slider("Expected Inflation Rate (%/year)", 1.0, 15.0, 6.0)

        nominal_fv = investment * (1 + expected_return / 100) ** years
        real_fv = investment * ((1 + expected_return / 100) / (1 + inflation_rate / 100)) ** years
        erosion = nominal_fv - real_fv

        st.metric("Nominal Future Value", f"₹{nominal_fv:,.0f}")
        st.metric("Real Future Value (Inflation-Adjusted)", f"₹{real_fv:,.0f}")
        st.metric("Purchasing Power Erosion", f"₹{erosion:,.0f}", delta=f"-{(erosion / nominal_fv) * 100:.1f}%")

        st.markdown("---")
        st.subheader("Key Insights")
        latest_cpi = float(series["CPI"].iloc[-1]) if len(series) > 0 else 0.0
        avg_yoy = float(series["YoY_Change"].dropna().mean()) if len(series) > 0 else 0.0
        st.markdown(f"""
        - **Latest CPI ({selected_category}):** {latest_cpi:.2f}
        - **Average YoY Inflation:** {avg_yoy:.2f}%
        - **Real Return:** {(expected_return - inflation_rate):.1f}% per year
        - **Rule of 72:** Money doubles in ~{72 / expected_return:.0f} years nominally,
          but only ~{72 / max(expected_return - inflation_rate, 0.1):.0f} years in real terms.
        """)

        st.info(
            "High inflation reduces real returns. Equity allocations with pricing power "
            "historically outperform fixed-income during inflationary periods."
        )
