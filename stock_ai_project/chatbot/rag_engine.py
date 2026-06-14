"""LangChain RAG engine for AI Stock Analyst chatbot."""

from __future__ import annotations

from langchain_core.documents import Document

from chatbot.prompts import ANALYSIS_TEMPLATE, BUY_QUERY_KEYWORDS, SYSTEM_PROMPT
from chatbot.vector_store import FinancialVectorStore
from models.anomaly_detection import AnomalyDetector
from models.sentiment_model import FinBERTSentimentAnalyzer
from models.xgboost_model import XGBoostStockModel
from utils.backtesting import run_backtest_report
from utils.data_loader import fetch_news, fetch_stock_history
from utils.feature_engineering import engineer_features, latest_feature_row, prepare_train_test
from utils.model_comparison import compare_models
from utils.volatility_regime import analyze_volatility_regime


class RAGEngine:
    """Retrieval-Augmented Generation engine for stock analysis queries."""

    def __init__(self):
        self.vector_store = FinancialVectorStore()
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.context_cache: dict = {}

    def _symbol_from_query(self, query: str) -> str | None:
        known = {
            "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infy": "INFY.NS",
            "infosys": "INFY.NS", "hdfc": "HDFCBANK.NS", "hdfcbank": "HDFCBANK.NS",
            "icici": "ICICIBANK.NS", "wipro": "WIPRO.NS", "itc": "ITC.NS",
            "apple": "AAPL", "microsoft": "MSFT", "msft": "MSFT",
        }
        q = query.lower()
        for key, sym in known.items():
            if key in q:
                return sym
        return None

    def build_context(self, symbol: str) -> dict:
        """Gather and index all intelligence for a symbol."""
        df = fetch_stock_history(symbol, period="2y")
        if df.empty:
            return {"error": f"No data found for {symbol}"}

        news = fetch_news(symbol.replace(".NS", "").replace(".BO", ""), num_articles=8)
        news_texts = [a.get("title", "") for a in news if a.get("title")]
        sentiment_results = self.sentiment_analyzer.analyze_batch(news_texts)
        sentiment_agg = self.sentiment_analyzer.aggregate_sentiment(sentiment_results)

        avg_sentiment = sentiment_agg.get("avg_score", 0.0)
        features = engineer_features(df, sentiment_score=avg_sentiment)
        train, test, feature_cols = prepare_train_test(features)

        prediction_summary = "Prediction unavailable — insufficient data."
        model_metrics = {}
        horizons = {}

        if not train.empty:
            xgb = XGBoostStockModel()
            xgb.fit(train[feature_cols], train["Target"])
            latest = latest_feature_row(features, feature_cols)
            base_price = float(df["Close"].iloc[-1])
            horizons = xgb.predict_horizons(latest, feature_cols, base_price)
            day1 = horizons.get(1, {})
            prediction_summary = (
                f"Next-day predicted price: {day1.get('predicted_price', 0):.2f} "
                f"({day1.get('change_pct', 0):+.2f}%). "
                f"7-day: {horizons.get(7, {}).get('predicted_price', 0):.2f}. "
                f"30-day: {horizons.get(30, {}).get('predicted_price', 0):.2f}."
            )

            comparison = compare_models(df, sentiment_score=avg_sentiment)
            model_metrics = comparison.get("metrics", {})

        prices = df.set_index("Date")["Close"] if "Date" in df.columns else df["Close"]
        vol_analysis = analyze_volatility_regime(prices)
        anomaly_df = self.anomaly_detector.fit_predict(df)
        anomaly_status = self.anomaly_detector.latest_status(anomaly_df)

        backtest = {}
        if not test.empty and "Target" in test.columns:
            backtest = run_backtest_report(
                test["Target"],
                test["Target"].shift(1).fillna(test["Target"].iloc[0]),
            )

        documents = []
        documents.append(Document(
            page_content=f"Stock {symbol} current close price is {df['Close'].iloc[-1]:.2f}.",
            metadata={"source": "market_data", "symbol": symbol},
        ))
        documents.append(Document(
            page_content=(
                f"FinBERT sentiment for {symbol}: mood={sentiment_agg.get('market_mood')}, "
                f"positive={sentiment_agg.get('positive_pct', 0):.1f}%, "
                f"negative={sentiment_agg.get('negative_pct', 0):.1f}%, "
                f"avg_score={avg_sentiment:.3f}."
            ),
            metadata={"source": "sentiment", "symbol": symbol},
        ))
        documents.append(Document(
            page_content=f"XGBoost prediction: {prediction_summary}",
            metadata={"source": "prediction", "symbol": symbol},
        ))
        documents.append(Document(
            page_content=(
                f"Risk: anomaly status={anomaly_status}, "
                f"volatility regime={vol_analysis.get('Regime')}, "
                f"annualized vol={vol_analysis.get('Current Volatility', 0):.2%}."
            ),
            metadata={"source": "risk", "symbol": symbol},
        ))

        for i, article in enumerate(news[:5]):
            title = article.get("title", "")
            if title:
                sr = sentiment_results[i] if i < len(sentiment_results) else None
                label = sr.label if sr else "neutral"
                documents.append(Document(
                    page_content=f"News ({label}): {title}",
                    metadata={"source": "news", "symbol": symbol},
                ))

        if backtest:
            documents.append(Document(
                page_content=(
                    f"Backtest: Sharpe={backtest.get('Sharpe Ratio', 0):.2f}, "
                    f"Max Drawdown={backtest.get('Max Drawdown', 0):.2f}%, "
                    f"Risk={backtest.get('Risk Recommendation', 'N/A')}."
                ),
                metadata={"source": "backtest", "symbol": symbol},
            ))

        self.vector_store.build_index(documents)

        context = {
            "symbol": symbol,
            "sentiment": sentiment_agg,
            "news": news,
            "prediction": horizons,
            "prediction_summary": prediction_summary,
            "volatility": vol_analysis,
            "anomaly_status": anomaly_status,
            "backtest": backtest,
            "model_metrics": model_metrics,
            "documents": documents,
        }
        self.context_cache[symbol] = context
        return context

    def _generate_recommendation(self, context: dict) -> str:
        sentiment = context.get("sentiment", {})
        mood = sentiment.get("market_mood", "Neutral")
        vol = context.get("volatility", {}).get("Regime", "Medium Volatility")
        anomaly = context.get("anomaly_status", "Normal")
        pred = context.get("prediction", {}).get(1, {})
        change = pred.get("change_pct", 0)

        if anomaly == "Anomaly":
            return "Exercise caution — anomaly detected. Wait for market stabilization before new positions."
        if mood == "Bullish" and change > 0 and "Low" in vol:
            return "Moderately favorable conditions. Consider gradual accumulation with strict risk controls."
        if mood == "Bearish" or change < -2:
            return "Unfavorable short-term outlook. Avoid aggressive buying; monitor for reversal signals."
        return "Mixed signals present. Conduct further due diligence before making investment decisions."

    def query(self, user_query: str, symbol: str | None = None) -> dict:
        """Process user query with RAG retrieval and structured response."""
        if symbol is None:
            symbol = self._symbol_from_query(user_query)

        if symbol is None:
            return {
                "answer": (
                    "Please specify a stock symbol (e.g., 'Should I buy Reliance?' or enter TCS.NS). "
                    "I analyze stocks using FinBERT sentiment, XGBoost predictions, and risk metrics."
                ),
                "sources": [],
                "symbol": None,
            }

        context = self.context_cache.get(symbol) or self.build_context(symbol)
        if "error" in context:
            return {"answer": context["error"], "sources": [], "symbol": symbol}

        retrieved = self.vector_store.search(user_query, k=5)
        source_texts = [doc.page_content for doc in retrieved]

        news_summary = "\n".join(f"• {s}" for s in source_texts if "News" in s) or "No recent news retrieved."
        sentiment = context["sentiment"]
        sentiment_summary = (
            f"Market Mood: {sentiment.get('market_mood')} | "
            f"Positive: {sentiment.get('positive_pct', 0):.1f}% | "
            f"Negative: {sentiment.get('negative_pct', 0):.1f}% | "
            f"Avg Score: {sentiment.get('avg_score', 0):.3f}"
        )
        risk_summary = (
            f"Anomaly: {context.get('anomaly_status')} | "
            f"Volatility: {context.get('volatility', {}).get('Regime')} | "
            f"{context.get('volatility', {}).get('Explanation', '')}"
        )
        recommendation = self._generate_recommendation(context)

        answer = ANALYSIS_TEMPLATE.format(
            symbol=symbol,
            sentiment_summary=sentiment_summary,
            news_summary=news_summary,
            prediction_summary=context.get("prediction_summary", "N/A"),
            risk_summary=risk_summary,
            recommendation=recommendation,
        )

        return {
            "answer": answer,
            "sources": source_texts,
            "symbol": symbol,
            "context": context,
            "system_prompt": SYSTEM_PROMPT,
        }
