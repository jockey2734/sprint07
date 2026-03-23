"""Sentiment-derived feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class SentimentFeatureBuilder:
    """Builds rolling sentiment features from daily sentiment DataFrame."""

    def __init__(self, rolling_window: int = 7, momentum_window: int = 3) -> None:
        self._rolling_window = rolling_window
        self._momentum_window = momentum_window

    def build(
        self,
        merged_df: pd.DataFrame,
        ko_sentiment_df: pd.DataFrame | None = None,
        en_sentiment_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Add sentiment rolling features to merged_df.

        Args:
            merged_df: Base merged DataFrame (price + raw sentiment).
            ko_sentiment_df: Korean-only daily sentiment (optional).
            en_sentiment_df: English-only daily sentiment (optional).

        Returns:
            DataFrame with additional sentiment feature columns.
        """
        if merged_df.empty:
            return merged_df

        result = merged_df.copy()

        if "compound" in result.columns:
            result = self._add_rolling_features(result, col="compound")

        # Korean / English divergence feature
        if ko_sentiment_df is not None and en_sentiment_df is not None:
            result = self._add_divergence_feature(result, ko_sentiment_df, en_sentiment_df)

        logger.debug("Sentiment features built")
        return result

    def _add_rolling_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add rolling mean, std, momentum for sentiment column."""
        result = df.copy()
        roll = result[col].rolling(self._rolling_window, min_periods=1)
        result[f"{col}_rolling_mean"] = roll.mean()
        result[f"{col}_rolling_std"] = roll.std().fillna(0)
        result[f"{col}_momentum"] = result[col].diff(self._momentum_window)
        # Z-score of sentiment within rolling window
        result[f"{col}_zscore"] = (
            (result[col] - result[f"{col}_rolling_mean"])
            / (result[f"{col}_rolling_std"] + 1e-9)
        )
        return result

    def _add_divergence_feature(
        self,
        df: pd.DataFrame,
        ko_df: pd.DataFrame,
        en_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute divergence between Korean and English sentiment."""
        result = df.copy()
        try:
            ko = ko_df["compound"].rename("ko_compound")
            en = en_df["compound"].rename("en_compound")
            aligned = pd.concat([ko, en], axis=1).reindex(result.index, method="ffill")
            result["sentiment_divergence"] = aligned["en_compound"] - aligned["ko_compound"]
        except Exception as exc:
            logger.warning(f"Divergence feature failed: {exc}")
        return result
