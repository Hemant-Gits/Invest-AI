# Academic Deliverables — InvestAI

This document supplements `README.md` with formal academic documentation for project submission and viva.

## 1. Problem Statement

Retail investors lack access to integrated AI tools that combine quantitative forecasting, sentiment analysis, explainability, and conversational research. Existing stock apps provide either basic charts or black-box predictions without transparency.

**InvestAI** solves this by building a modular decision support system that merges ML/DL prediction, FinBERT NLP, SHAP explainability, and RAG-based financial intelligence.

## 2. Objectives

| # | Objective | Implementation |
|---|-----------|----------------|
| 1 | Accurate multi-horizon stock prediction | XGBoost with 18+ engineered features |
| 2 | Model benchmarking | XGBoost vs LSTM vs Linear Regression |
| 3 | Transparent AI decisions | SHAP global + local explanations |
| 4 | Finance-specific sentiment | FinBERT replacing TextBlob |
| 5 | Research contribution | Price-only vs price+sentiment comparison |
| 6 | Risk management | Anomaly detection, volatility regime, portfolio analytics |
| 7 | Conversational research | RAG chatbot with FAISS + LangChain |
| 8 | Professional reporting | ReportLab PDF generation |

## 3. Technology Justification

| Technology | Justification |
|------------|---------------|
| **Streamlit** | Rapid prototyping of data-heavy dashboards; ideal for academic demos |
| **XGBoost** | State-of-the-art gradient boosting for tabular financial features |
| **LSTM** | Captures sequential temporal dependencies in price series |
| **FinBERT** | Domain-specific pre-training on financial corpora |
| **SHAP** | Model-agnostic Shapley values for explainability |
| **FAISS** | Efficient similarity search for RAG retrieval at scale |
| **LangChain** | Standardized RAG pipeline orchestration |
| **Isolation Forest** | Effective unsupervised anomaly detection |
| **yfinance** | Free, reliable OHLCV data for NSE/BSE and global markets |

## 4. Algorithms Used

### Feature Engineering
- Lag features (1, 3, 5, 10 days)
- Rolling means (5, 10, 20 days)
- RSI (14-period), MACD (12/26/9), Bollinger Bands (20, 2σ)
- Rolling volatility (20-day)

### XGBoost Hyperparameters
- n_estimators=200, max_depth=6, learning_rate=0.05
- subsample=0.8, colsample_bytree=0.8

### LSTM Architecture
- LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)
- Sequence length: 10, Epochs: 15

### Evaluation Metrics
- RMSE, MAE, MAPE, R² Score, Directional Accuracy

## 5. Testing Strategy

| Test Type | Method |
|-----------|--------|
| Unit | Feature engineering output shape and NaN handling |
| Integration | End-to-end pipeline: data → features → model → prediction |
| Model | Chronological train/test split (no data leakage) |
| UI | Manual testing of all 6 Streamlit pages |
| RAG | Query retrieval relevance verification |

## 6. Results Interpretation Guide

- **RMSE/MAE**: Lower is better; measures prediction error in price units
- **Directional Accuracy**: % of correct up/down predictions; critical for trading signals
- **Sharpe Ratio**: >1.0 indicates good risk-adjusted returns
- **Sentiment Improvement %**: Research metric — positive means FinBERT adds value
- **Diversification Score**: Higher = lower average correlation between holdings

## 7. Limitations

1. yfinance data may have delays; not suitable for high-frequency trading
2. FinBERT first load downloads ~400MB model weights
3. LSTM training is computationally intensive on CPU
4. NewsAPI free tier has request limits
5. Predictions are educational — real markets have unforeseeable events

## 8. Conclusion

InvestAI demonstrates a production-quality architecture for AI-driven financial analytics suitable for final-year evaluation. It integrates six AI disciplines (ML, DL, NLP, XAI, RAG, anomaly detection) in a single cohesive platform with clear academic research contribution around sentiment-enhanced prediction.
