"""XGBoost stock prediction model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class XGBoostStockModel:
    """XGBoost regressor for next-day and multi-day stock price forecasting."""

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )
        self.residual_std: float = 0.0

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostStockModel":
        self.model.fit(x_train, y_train)
        preds = self.model.predict(x_train)
        self.residual_std = float(np.std(y_train.values - preds))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x)

    def predict_horizons(
        self,
        latest_features: pd.DataFrame,
        feature_cols: list[str],
        base_price: float,
        horizons: list[int] | None = None,
    ) -> dict[int, dict]:
        """Iterative multi-step forecasting with confidence bands."""
        if horizons is None:
            horizons = [1, 7, 30]

        results = {}
        current = latest_features[feature_cols].copy()
        price = base_price

        max_horizon = max(horizons)
        path = []

        for day in range(1, max_horizon + 1):
            pred = float(self.model.predict(current)[0])
            lower = pred - 1.96 * self.residual_std
            upper = pred + 1.96 * self.residual_std
            path.append({"day": day, "price": pred, "lower": lower, "upper": upper})

            if day in horizons:
                change_pct = ((pred - base_price) / base_price) * 100 if base_price else 0
                results[day] = {
                    "predicted_price": pred,
                    "change_pct": change_pct,
                    "lower": lower,
                    "upper": upper,
                    "confidence": max(0, min(100, 100 - abs(change_pct))),
                }

            if "Lag_1" in current.columns:
                current["Lag_1"] = pred
            price = pred

        results["path"] = path
        return results

    def feature_importance(self) -> pd.DataFrame:
        importances = self.model.feature_importances_
        names = self.model.feature_names_in_
        return pd.DataFrame({"Feature": names, "Importance": importances}).sort_values(
            "Importance", ascending=False
        )
