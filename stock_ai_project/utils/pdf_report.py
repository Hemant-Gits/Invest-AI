"""PDF report generation using ReportLab."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils.config import REPORTS_DIR


def generate_stock_report(
    stock_name: str,
    prediction: dict,
    sentiment: dict,
    anomaly_status: str,
    volatility: dict,
    backtest: dict,
    model_metrics: dict,
    portfolio: dict | None = None,
) -> Path:
    """Generate professional PDF investment analysis report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = REPORTS_DIR / f"{stock_name.replace('.', '_')}_report_{timestamp}.pdf"

    doc = SimpleDocTemplate(str(filepath), pagesize=A4, topMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], textColor=colors.HexColor("#4f46e5"))
    heading = styles["Heading2"]
    body = styles["BodyText"]
    story = []

    story.append(Paragraph("InvestAI — Stock Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", body))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph(f"Stock: {stock_name}", heading))
    story.append(Spacer(1, 0.15 * inch))

    pred_data = [
        ["Horizon", "Predicted Price", "Change %", "Confidence Interval"],
    ]
    for horizon, data in prediction.items():
        if horizon == "path":
            continue
        pred_data.append([
            f"{horizon}-Day",
            f"{data.get('predicted_price', 0):.2f}",
            f"{data.get('change_pct', 0):+.2f}%",
            f"[{data.get('lower', 0):.2f} – {data.get('upper', 0):.2f}]",
        ])

    pred_table = Table(pred_data, colWidths=[1.2 * inch, 1.5 * inch, 1.2 * inch, 2 * inch])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Paragraph("Price Predictions", heading))
    story.append(pred_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Sentiment Analysis (FinBERT)", heading))
    story.append(Paragraph(
        f"Market Mood: {sentiment.get('market_mood', 'N/A')} | "
        f"Avg Score: {sentiment.get('avg_score', 0):.3f} | "
        f"Positive: {sentiment.get('positive_pct', 0):.1f}% | "
        f"Negative: {sentiment.get('negative_pct', 0):.1f}%",
        body,
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Risk Indicators", heading))
    story.append(Paragraph(f"Anomaly Status: {anomaly_status}", body))
    story.append(Paragraph(
        f"Volatility Regime: {volatility.get('Regime', 'N/A')} "
        f"({volatility.get('Current Volatility', 0):.2%} annualized)",
        body,
    ))
    story.append(Spacer(1, 0.15 * inch))

    if backtest:
        story.append(Paragraph("Backtest Metrics", heading))
        bt_rows = [["Metric", "Value"]]
        for key in ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate", "CAGR", "Total Return", "Risk Recommendation"]:
            if key in backtest:
                val = backtest[key]
                bt_rows.append([key, f"{val:.2f}" if isinstance(val, float) else str(val)])
        bt_table = Table(bt_rows, colWidths=[2.5 * inch, 2 * inch])
        bt_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecf0f1")),
        ]))
        story.append(bt_table)
        story.append(Spacer(1, 0.15 * inch))

    if model_metrics:
        story.append(Paragraph("Model Performance", heading))
        for model_name, metrics in model_metrics.items():
            story.append(Paragraph(
                f"{model_name}: RMSE={metrics.get('RMSE', 0):.4f}, "
                f"MAE={metrics.get('MAE', 0):.4f}, "
                f"Directional Accuracy={metrics.get('Directional Accuracy', 0):.1f}%",
                body,
            ))

    if portfolio:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("Portfolio Summary", heading))
        story.append(Paragraph(
            f"Return: {portfolio.get('Portfolio Return', 0):.2f}% | "
            f"Volatility: {portfolio.get('Portfolio Volatility', 0):.2f}% | "
            f"Diversification: {portfolio.get('Diversification Score', 0):.1f}/100",
            body,
        ))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "Disclaimer: This report is for academic and research purposes only. "
        "Not financial advice.",
        ParagraphStyle("Disclaimer", parent=body, fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    return filepath
