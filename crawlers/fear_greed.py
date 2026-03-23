"""CNN Fear & Greed Index crawler."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import requests
from loguru import logger

from stock_ai.crawlers.base import BaseCrawler, FearGreedData, RawPost
from stock_ai.storage.cache import FileCache

_API_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://edition.cnn.com/",
}

_LABEL_MAP = {
    (0, 25): "Extreme Fear",
    (25, 45): "Fear",
    (45, 55): "Neutral",
    (55, 75): "Greed",
    (75, 101): "Extreme Greed",
}


def _value_to_label(value: float) -> str:
    for (lo, hi), label in _LABEL_MAP.items():
        if lo <= value < hi:
            return label
    return "Unknown"


class FearGreedCrawler(BaseCrawler):
    """Fetches CNN Fear & Greed Index data."""

    def __init__(self, cache: Optional[FileCache] = None) -> None:
        self._cache = cache or FileCache()

    def is_available(self) -> bool:
        return True

    def crawl(self, ticker: str, **kwargs) -> tuple[RawPost, ...]:
        """Not applicable — use fetch() instead. Returns empty tuple."""
        return ()

    def fetch(self) -> Optional[FearGreedData]:
        """Fetch current Fear & Greed index. Returns None on failure."""
        cache_key = "fear_greed:current"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            resp = requests.get(_API_URL, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            fg_data = data.get("fear_and_greed", {})
            current_score = float(fg_data.get("score", 50))
            prev_week = fg_data.get("previous_week", {})
            prev_month = fg_data.get("previous_month", {})

            result = FearGreedData(
                value=current_score,
                label=_value_to_label(current_score),
                collected_at=datetime.utcnow(),
                one_week_ago=float(prev_week.get("score", 0)) if prev_week else None,
                one_month_ago=float(prev_month.get("score", 0)) if prev_month else None,
            )
            self._cache.set(cache_key, result, ttl=3600)
            logger.info(f"Fear & Greed: {result.value:.1f} ({result.label})")
            return result
        except Exception as exc:
            logger.warning(f"Fear & Greed fetch failed: {exc}")
            return None
