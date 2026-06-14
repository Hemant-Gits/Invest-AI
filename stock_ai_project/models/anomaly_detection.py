"""Isolation Forest anomaly detection for market data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Detect price crashes, volume spikes, and unusual patterns."""

    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        data = df.copy()
        if "Date" in data.columns:
            data = data.set_index("Date")

        features = pd.DataFrame(index=data.index)
        features["Return"] = data["Close"].pct_change()
        features["Volume_Change"] = data["Volume"].pct_change()
        features["Price_Range"] = (data["High"] - data["Low"]) / data["Close"]
        features["Return_Vol"] = features["Return"].rolling(5).std()
        features = features.replace([np.inf, -np.inf], np.nan).dropna()

        if len(features) < 20:
            return pd.DataFrame()

        preds = self.model.fit_predict(features)
        scores = self.model.decision_function(features)

        result = features.copy()
        result["Anomaly"] = preds
        result["Anomaly_Score"] = scores
        result["Status"] = result["Anomaly"].apply(
            lambda x: "Anomaly" if x == -1 else "Normal"
        )

        warning_mask = (result["Anomaly_Score"] < 0) & (result["Anomaly"] == 1)
        result.loc[warning_mask, "Status"] = "Warning"
        result = result.reset_index()
        return result

    def latest_status(self, anomaly_df: pd.DataFrame) -> str:
        if anomaly_df.empty:
            return "Normal"
        return str(anomaly_df["Status"].iloc[-1])

    def count_anomalies(self, anomaly_df: pd.DataFrame) -> int:
        if anomaly_df.empty:
            return 0
        return int((anomaly_df["Status"] == "Anomaly").sum())
