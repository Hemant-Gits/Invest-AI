"""Prompt templates for RAG chatbot."""

SYSTEM_PROMPT = """You are InvestAI, an intelligent financial analyst assistant.
Answer using ONLY the provided context from news, sentiment, predictions, and risk metrics.
Be clear, structured, and suitable for investment decision support.
Always mention uncertainty and that this is not financial advice."""

ANALYSIS_TEMPLATE = """
Based on the retrieved financial intelligence for {symbol}:

## Market Sentiment
{sentiment_summary}

## Latest News Highlights
{news_summary}

## AI Model Prediction
{prediction_summary}

## Risk Assessment
{risk_summary}

## Analyst Recommendation
{recommendation}

---
Note: This analysis combines FinBERT sentiment, XGBoost predictions, and quantitative risk metrics.
For educational/research purposes only — not financial advice.
"""

BUY_QUERY_KEYWORDS = ["buy", "invest", "should i", "worth", "good time", "purchase", "hold", "sell"]
