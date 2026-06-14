"""Model evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def directional_accuracy(y_true, y_pred) -> float:
    """Percentage of correct up/down direction predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return 0.0
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(actual_dir == pred_dir) * 100)


def compute_all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "Directional Accuracy": directional_accuracy(y_true, y_pred),
    }


def rank_models(metrics_dict: dict[str, dict[str, float]]) -> list[tuple[str, float]]:
    """Rank models by composite score (lower error + higher R2 + higher direction accuracy)."""
    scores = {}
    for name, metrics in metrics_dict.items():
        score = (
            metrics.get("RMSE", 999)
            + metrics.get("MAE", 999)
            + metrics.get("MAPE", 999) * 0.01
            - metrics.get("R2", 0) * 10
            - metrics.get("Directional Accuracy", 0) * 0.05
        )
        scores[name] = score
    return sorted(scores.items(), key=lambda x: x[1])
