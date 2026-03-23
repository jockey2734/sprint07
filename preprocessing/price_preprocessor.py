"""Price data preprocessing: missing values, normalization, LSTM sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class LSTMSequenceData:
    """Prepared LSTM input sequences."""

    X: np.ndarray   # shape (samples, seq_len, features)
    y: np.ndarray   # shape (samples,)
    scaler: MinMaxScaler
    feature_cols: tuple[str, ...]


class PricePreprocessor:
    """Preprocesses raw OHLCV price DataFrames for model training."""

    def __init__(self, target_col: str = "Close") -> None:
        self._target_col = target_col

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values and add log return column."""
        if df.empty:
            return df
        result = df.copy()
        # Forward fill then backward fill
        result = result.ffill().bfill()
        result["log_return"] = np.log(result[self._target_col] / result[self._target_col].shift(1))
        result["pct_change"] = result[self._target_col].pct_change()
        result = result.dropna(subset=["log_return"])
        return result

    def normalize(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
    ) -> tuple[pd.DataFrame, MinMaxScaler]:
        """Min-max normalize feature columns. Returns (scaled_df, scaler)."""
        cols = feature_cols or [c for c in df.columns if c != self._target_col]
        cols = [c for c in cols if c in df.columns]
        scaler = MinMaxScaler()
        result = df.copy()
        result[cols] = scaler.fit_transform(df[cols])
        return result, scaler

    def create_lstm_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_length: int = 60,
        horizon: int = 1,
    ) -> LSTMSequenceData:
        """Create overlapping sliding-window sequences for LSTM.

        Args:
            df: Preprocessed price DataFrame.
            feature_cols: Input feature column names.
            seq_length: Look-back window size.
            horizon: Days ahead to predict.

        Returns:
            LSTMSequenceData with X (N, seq_len, F), y (N,) arrays.
        """
        valid_cols = [c for c in feature_cols if c in df.columns]
        if not valid_cols:
            raise ValueError("No valid feature columns found in DataFrame")

        data = df[valid_cols].values  # (T, F)
        target = df[self._target_col].values  # (T,)

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        X_list: list[np.ndarray] = []
        y_list: list[float] = []

        for i in range(seq_length, len(data_scaled) - horizon + 1):
            X_list.append(data_scaled[i - seq_length : i])
            y_list.append(target[i + horizon - 1])

        if not X_list:
            raise ValueError(
                f"Not enough data for sequences: need >{seq_length + horizon} rows, got {len(df)}"
            )

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        logger.info(f"LSTM sequences: X={X.shape}, y={y.shape}")
        return LSTMSequenceData(X=X, y=y, scaler=scaler, feature_cols=tuple(valid_cols))
