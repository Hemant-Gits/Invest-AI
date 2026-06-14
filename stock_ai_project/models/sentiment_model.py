"""FinBERT financial sentiment analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SentimentResult:
    label: str
    score: float
    confidence: float
    polarity: float


class FinBERTSentimentAnalyzer:
    """Finance-specific sentiment classifier using ProsusAI/finbert."""

    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self):
        self.pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
                max_length=512,
            )
        except Exception:
            self.pipeline = None

    def _fallback_analyze(self, text: str) -> SentimentResult:
        """Keyword-based fallback when FinBERT is unavailable."""
        text_lower = text.lower()
        positive_words = ["gain", "rise", "profit", "growth", "bull", "up", "strong", "beat", "surge"]
        negative_words = ["loss", "fall", "decline", "bear", "down", "weak", "miss", "crash", "risk"]
        pos = sum(w in text_lower for w in positive_words)
        neg = sum(w in text_lower for w in negative_words)
        if pos > neg:
            return SentimentResult("positive", 0.6, 0.55, 0.6)
        if neg > pos:
            return SentimentResult("negative", -0.6, 0.55, -0.6)
        return SentimentResult("neutral", 0.0, 0.5, 0.0)

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult("neutral", 0.0, 0.0, 0.0)

        if self.pipeline is None:
            return self._fallback_analyze(text)

        try:
            result = self.pipeline(text[:512])[0]
            label = result["label"].lower()
            confidence = float(result["score"])
            polarity = confidence if label == "positive" else (-confidence if label == "negative" else 0.0)
            return SentimentResult(label, polarity, confidence, polarity)
        except Exception:
            return self._fallback_analyze(text)

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.analyze(t) for t in texts]

    def aggregate_sentiment(self, results: list[SentimentResult]) -> dict:
        if not results:
            return {"positive_pct": 0, "negative_pct": 0, "neutral_pct": 0, "avg_score": 0, "market_mood": "Neutral"}

        total = len(results)
        pos = sum(1 for r in results if r.label == "positive")
        neg = sum(1 for r in results if r.label == "negative")
        neu = total - pos - neg
        avg_score = sum(r.polarity for r in results) / total

        if pos > neg and avg_score > 0.1:
            mood = "Bullish"
        elif neg > pos and avg_score < -0.1:
            mood = "Bearish"
        else:
            mood = "Neutral"

        return {
            "positive_pct": pos / total * 100,
            "negative_pct": neg / total * 100,
            "neutral_pct": neu / total * 100,
            "avg_score": avg_score,
            "market_mood": mood,
            "results": results,
        }
