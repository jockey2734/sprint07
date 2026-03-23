"""Final feature assembly: combines technical + sentiment + fundamental features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureDataset:
    """Assembled training dataset ready for model consumption."""

    X: pd.DataFrame          # Feature matrix
    y: pd.Series             # Target: next-day return (pct_change)
    feature_names: tuple[str, ...]
    scaler: Optional[StandardScaler]
    metadata: dict           # ticker, date_range, n_samples


# Columns to EXCLUDE from features (raw prices, redundant, etc.)
_EXCLUDE_COLS = frozenset([
    "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
    "Adj Close", "adj_close", "log_return", "pct_change",
    "source", "label",
])


class FeatureBuilder:
    """Assembles (X, y) training dataset from merged + enriched DataFrame."""

    def __init__(self, horizon_days: int = 1, scale: bool = True) -> None:
        self._horizon = horizon_days
        self._scale = scale

    def build(
        self,
        merged_df: pd.DataFrame,
        ticker: str = "unknown",
    ) -> FeatureDataset:
        """Build final (X, y) dataset.

        Args:
            merged_df: DataFrame with price + technical + sentiment + fundamental columns.
            ticker: Ticker name for metadata.

        Returns:
            FeatureDataset with immutable feature_names tuple.
        """
        if merged_df.empty:
            raise ValueError("Cannot build features from empty DataFrame")

        df = merged_df.copy()

        # Build target: forward return over horizon days
        df["_target"] = df["Close"].pct_change(self._horizon).shift(-self._horizon)
        df = df.dropna(subset=["_target"])

        # Select feature columns
        feature_cols = [
            c for c in df.columns
            if c not in _EXCLUDE_COLS and c != "_target" and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        X_raw = df[feature_cols].copy()
        y = df["_target"].copy()

        # Drop rows where X has all NaN
        valid_mask = X_raw.notna().any(axis=1)
        X_raw = X_raw[valid_mask]
        y = y[valid_mask]

        # Fill remaining NaN with column median
        X_filled = X_raw.fillna(X_raw.median())

        scaler: Optional[StandardScaler] = None
        if self._scale and not X_filled.empty:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_filled),
                index=X_filled.index,
                columns=X_filled.columns,
            )
        else:
            X_scaled = X_filled

        metadata = {
            "ticker": ticker,
            "date_range": f"{df.index.min().date()} → {df.index.max().date()}",
            "n_samples": len(X_scaled),
            "horizon_days": self._horizon,
            "n_features": len(feature_cols),
        }

        logger.info(
            f"FeatureBuilder: {len(X_scaled)} samples × {len(feature_cols)} features "
            f"(horizon={self._horizon}d) for {ticker}"
        )
        return FeatureDataset(
            X=X_scaled,
            y=y,
            feature_names=tuple(feature_cols),
            scaler=scaler,
            metadata=metadata,
        )
