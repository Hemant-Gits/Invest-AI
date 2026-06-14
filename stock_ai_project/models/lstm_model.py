"""LSTM deep learning model for stock prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class LSTMStockModel:
    """Sequence-based LSTM regressor using TensorFlow/Keras."""

    def __init__(self, sequence_length: int = 10, epochs: int = 15, batch_size: int = 16):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_cols: list[str] = []

    def _build_sequences(self, data: np.ndarray, targets: np.ndarray):
        x, y = [], []
        for i in range(self.sequence_length, len(data)):
            x.append(data[i - self.sequence_length : i])
            y.append(targets[i])
        return np.array(x), np.array(y)

    def train_and_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict | None:
        try:
            import tensorflow as tf
            from tensorflow.keras.layers import Dense, Dropout, LSTM
            from tensorflow.keras.models import Sequential
        except ImportError:
            return None

        self.feature_cols = feature_cols
        train_x = train[feature_cols].values
        train_y = train["Target"].values
        test_x = test[feature_cols].values
        test_y = test["Target"].values

        combined_x = np.vstack([train_x, test_x])
        combined_y = np.concatenate([train_y, test_y])
        scaled_x = self.scaler.fit_transform(combined_x)

        x_seq, y_seq = self._build_sequences(scaled_x, combined_y)
        if len(x_seq) < 20:
            return None

        split = len(train) - self.sequence_length
        if split <= 0:
            split = int(len(x_seq) * 0.8)

        x_train, y_train = x_seq[:split], y_seq[:split]
        x_test, y_test = x_seq[split:], y_seq[split:]

        if len(x_test) == 0:
            return None

        tf.keras.utils.set_random_seed(42)
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, len(feature_cols))),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(
            x_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1,
        )

        y_pred = self.model.predict(x_test, verbose=0).flatten()
        return {"y_test": y_test, "y_pred": y_pred}
