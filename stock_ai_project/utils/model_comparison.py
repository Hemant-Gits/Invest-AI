"""Unified model comparison framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.lstm_model import LSTMStockModel
from models.xgboost_model import XGBoostStockModel
from utils.feature_engineering import engineer_features, prepare_train_test
from utils.metrics import compute_all_metrics, rank_models


def compare_models(
    df: pd.DataFrame,
    sentiment_score: float | None = None,
) -> dict:
    """Train XGBoost, LSTM, and Linear Regression on identical splits."""
    features = engineer_features(df, sentiment_score=sentiment_score)
    train, test, feature_cols = prepare_train_test(features)
    if train.empty or test.empty:
        return {}

    x_train, y_train = train[feature_cols], train["Target"]
    x_test, y_test = test[feature_cols], test["Target"]

    results = {}
    predictions = {}

    xgb = XGBoostStockModel()
    xgb.fit(x_train, y_train)
    xgb_pred = xgb.predict(x_test)
    results["XGBoost"] = compute_all_metrics(y_test, xgb_pred)
    predictions["XGBoost"] = xgb_pred

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_pred = lr.predict(x_test)
    results["Linear Regression"] = compute_all_metrics(y_test, lr_pred)
    predictions["Linear Regression"] = lr_pred

    lstm = LSTMStockModel(sequence_length=10)
    lstm_result = lstm.train_and_predict(train, test, feature_cols)
    if lstm_result:
        results["LSTM"] = compute_all_metrics(lstm_result["y_test"], lstm_result["y_pred"])
        predictions["LSTM"] = lstm_result["y_pred"]
    else:
        results["LSTM"] = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan, "Directional Accuracy": np.nan}

    ranking = rank_models({k: v for k, v in results.items() if not np.isnan(v.get("RMSE", np.nan))})
    best_model = ranking[0][0] if ranking else "XGBoost"

    return {
        "metrics": results,
        "predictions": predictions,
        "y_test": y_test.values,
        "test_dates": test["Date"].values if "Date" in test.columns else None,
        "ranking": ranking,
        "best_model": best_model,
        "xgb_model": xgb,
        "feature_cols": feature_cols,
        "train": train,
        "test": test,
    }


def compare_sentiment_research(df: pd.DataFrame, sentiment_score: float) -> dict:
    """Research contribution: Model A (price only) vs Model B (price + sentiment)."""
    base = compare_models(df, sentiment_score=None)
    enhanced = compare_models(df, sentiment_score=sentiment_score)

    if not base or not enhanced:
        return {}

    base_rmse = base["metrics"]["XGBoost"]["RMSE"]
    enhanced_rmse = enhanced["metrics"]["XGBoost"]["RMSE"]
    improvement = 0.0
    if base_rmse > 0:
        improvement = ((base_rmse - enhanced_rmse) / base_rmse) * 100

    return {
        "model_a": base["metrics"]["XGBoost"],
        "model_b": enhanced["metrics"]["XGBoost"],
        "improvement_pct": improvement,
        "summary": f"Sentiment improved prediction accuracy by {improvement:.2f}% (RMSE reduction).",
    }
