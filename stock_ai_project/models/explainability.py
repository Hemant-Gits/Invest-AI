"""SHAP-based explainable AI utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


FEATURE_LABELS = {
    "RSI": "RSI momentum signal",
    "MACD": "MACD trend indicator",
    "Volatility": "Market volatility",
    "Volume": "Trading volume",
    "Sentiment_Score": "News sentiment score",
    "Lag_1": "Previous day price",
    "Rolling_Mean_5": "Short-term trend",
    "BB_Upper": "Bollinger upper band",
    "BB_Lower": "Bollinger lower band",
}


def explain_prediction(
    model,
    x_train: pd.DataFrame,
    x_instance: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Generate global and local SHAP explanations."""
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(x_train[feature_cols])

    instance_values = explainer.shap_values(x_instance[feature_cols])[0]
    feature_impact = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP Value": instance_values,
        "Feature Value": x_instance[feature_cols].iloc[0].values,
    }).sort_values("SHAP Value", key=abs, ascending=False)

    positive = feature_impact[feature_impact["SHAP Value"] > 0].head(3)
    negative = feature_impact[feature_impact["SHAP Value"] < 0].head(3)

    def format_reason(row):
        label = FEATURE_LABELS.get(row["Feature"], row["Feature"])
        direction = "supports rise" if row["SHAP Value"] > 0 else "pressures decline"
        return f"{label} {direction}"

    positive_reasons = [format_reason(r) for _, r in positive.iterrows()]
    negative_reasons = [format_reason(r) for _, r in negative.iterrows()]

    fig_summary = _create_summary_plot(shap_values, x_train[feature_cols])
    fig_importance = _create_importance_plot(feature_impact)
    fig_waterfall = _create_waterfall_plot(explainer, x_instance[feature_cols])

    return {
        "feature_impact": feature_impact,
        "positive_reasons": positive_reasons,
        "negative_reasons": negative_reasons,
        "fig_summary": fig_summary,
        "fig_importance": fig_importance,
        "fig_waterfall": fig_waterfall,
        "shap_values": shap_values,
        "explainer": explainer,
    }


def _create_summary_plot(shap_values, x_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, x_data, show=False, plot_type="bar")
    plt.tight_layout()
    return fig


def _create_importance_plot(feature_impact: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    top = feature_impact.head(10)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top["SHAP Value"]]
    ax.barh(top["Feature"], top["SHAP Value"], color=colors)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title("Feature Importance for This Prediction")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def _create_waterfall_plot(explainer, x_instance):
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        shap_values = explainer(x_instance)
        shap.plots.waterfall(shap_values[0], show=False)
    except Exception:
        ax.text(0.5, 0.5, "Waterfall plot unavailable", ha="center", va="center")
    plt.tight_layout()
    return fig


def human_explanation(change_pct: float, explanation: dict) -> str:
    direction = "Rise" if change_pct >= 0 else "Fall"
    lines = [f"Predicted {direction} = {change_pct:+.2f}%", "", "Main Reasons:"]
    for reason in explanation.get("positive_reasons", [])[:3]:
        lines.append(f"  • {reason}")
    lines.append("")
    lines.append("Negative Contributors:")
    for reason in explanation.get("negative_reasons", [])[:3]:
        lines.append(f"  • {reason}")
    return "\n".join(lines)
