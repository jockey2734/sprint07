"""Date-based LEFT JOIN merger: price + sentiment + Fear&Greed + fundamentals."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from stock_ai.crawlers.base import FearGreedData


class DataMerger:
    """Merges price, sentiment, and Fear & Greed data on date index."""

    def merge(
        self,
        price_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None,
        fear_greed: Optional[FearGreedData] = None,
        fundamental_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Perform LEFT JOIN of all data sources on date index.

        Args:
            price_df: OHLCV DataFrame indexed by date.
            sentiment_df: Daily sentiment DataFrame indexed by date.
            fear_greed: Current Fear & Greed snapshot (broadcasted).
            fundamental_df: Quarterly fundamentals (forward-filled).

        Returns:
            Merged DataFrame indexed by date.
        """
        if price_df.empty:
            logger.warning("DataMerger: price_df is empty — cannot merge")
            return pd.DataFrame()

        result = price_df.copy()
        result.index = pd.to_datetime(result.index).normalize()

        # Merge sentiment (LEFT JOIN on date)
        if sentiment_df is not None and not sentiment_df.empty:
            sent = sentiment_df.copy()
            sent.index = pd.to_datetime(sent.index).normalize()
            # Select sentiment columns, prefix if needed
            sent_cols = [c for c in sent.columns if c not in result.columns]
            sent_renamed = sent[sent_cols] if sent_cols else sent
            result = result.join(sent_renamed, how="left")
            logger.debug(f"Merged sentiment: {sent_renamed.shape[1]} columns")

        # Broadcast Fear & Greed as scalar columns
        if fear_greed is not None:
            result["fg_value"] = fear_greed.value
            result["fg_one_week_ago"] = fear_greed.one_week_ago
            result["fg_one_month_ago"] = fear_greed.one_month_ago
            result["fg_delta_week"] = (
                fear_greed.value - fear_greed.one_week_ago
                if fear_greed.one_week_ago is not None
                else None
            )
            logger.debug("Merged Fear & Greed index")

        # Merge fundamentals (quarterly → forward fill to daily)
        if fundamental_df is not None and not fundamental_df.empty:
            fund = fundamental_df.copy()
            fund.index = pd.to_datetime(fund.index).normalize()
            # Reindex to price dates and forward-fill quarterly data
            fund_reindexed = fund.reindex(result.index, method="ffill")
            fund_cols = [c for c in fund_reindexed.columns if c not in result.columns]
            if fund_cols:
                result = result.join(fund_reindexed[fund_cols], how="left")
                logger.debug(f"Merged fundamentals: {len(fund_cols)} columns")

        # Forward fill any NaN from left joins
        result = result.ffill()
        logger.info(f"DataMerger: result shape {result.shape}")
        return result
