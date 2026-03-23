"""Daily sentiment aggregation, weighting, and spike detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
from loguru import logger

from stock_ai.crawlers.base import RawPost
from stock_ai.sentiment.base_analyzer import SentimentResult


@dataclass(frozen=True)
class SentimentSpike:
    """Detected sudden change in sentiment."""

    date: pd.Timestamp
    before: float
    after: float
    delta: float
    direction: str  # "bullish_spike" | "bearish_spike"


class SentimentAggregator:
    """Aggregates per-post sentiment results into daily time series."""

    def __init__(self, spike_threshold: float = 0.3) -> None:
        self._spike_threshold = spike_threshold

    def aggregate(
        self,
        posts: Sequence[RawPost],
        sentiments: Sequence[SentimentResult],
        source_filter: str | None = None,
    ) -> pd.DataFrame:
        """Combine posts + sentiment results into a daily DataFrame.

        Returns DataFrame indexed by date with columns:
          positive, negative, neutral, compound, weighted_score,
          post_count, avg_upvotes, source
        """
        if len(posts) != len(sentiments):
            raise ValueError(
                f"posts ({len(posts)}) and sentiments ({len(sentiments)}) length mismatch"
            )

        records: list[dict] = []
        for post, sent in zip(posts, sentiments):
            if source_filter and post.source != source_filter:
                continue
            weight = max(1, post.upvotes)  # upvote-weighted
            records.append(
                {
                    "date": post.created_at.date(),
                    "positive": sent.positive,
                    "negative": sent.negative,
                    "neutral": sent.neutral,
                    "compound": sent.compound,
                    "weight": weight,
                    "source": post.source,
                }
            )

        if not records:
            logger.warning("SentimentAggregator: no records to aggregate")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        # Weighted aggregation by day
        grouped = df.groupby("date").apply(self._weighted_agg).reset_index()
        grouped = grouped.set_index("date").sort_index()
        logger.info(f"Sentiment aggregated: {len(grouped)} daily rows")
        return grouped

    def _weighted_agg(self, group: pd.DataFrame) -> pd.Series:
        w = group["weight"].values
        total_w = w.sum() or 1.0
        return pd.Series(
            {
                "positive": float(np.average(group["positive"], weights=w)),
                "negative": float(np.average(group["negative"], weights=w)),
                "neutral": float(np.average(group["neutral"], weights=w)),
                "compound": float(np.average(group["compound"], weights=w)),
                "weighted_score": float(np.average(group["compound"], weights=w)),
                "post_count": len(group),
                "avg_upvotes": float(group["weight"].mean()),
                "source": group["source"].mode().iloc[0] if not group.empty else "unknown",
            }
        )

    def detect_spikes(
        self, daily_df: pd.DataFrame, window: int = 3
    ) -> tuple[SentimentSpike, ...]:
        """Detect sudden sentiment shifts using rolling mean delta."""
        if "compound" not in daily_df.columns or len(daily_df) < window + 1:
            return ()

        roll = daily_df["compound"].rolling(window).mean()
        delta = roll.diff()
        spikes: list[SentimentSpike] = []

        for idx in daily_df.index:
            d = delta.get(idx, 0.0)
            if abs(d) >= self._spike_threshold:
                prev = roll.shift(1).get(idx, 0.0)
                curr = roll.get(idx, 0.0)
                spikes.append(
                    SentimentSpike(
                        date=idx,
                        before=float(prev),
                        after=float(curr),
                        delta=float(d),
                        direction="bullish_spike" if d > 0 else "bearish_spike",
                    )
                )
        logger.info(f"Sentiment spikes detected: {len(spikes)}")
        return tuple(spikes)
