"""Base crawler abstractions: RawPost frozen dataclass and BaseCrawler ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RawPost:
    """Immutable container for a single crawled post/article."""

    source: str          # e.g. "naver", "reddit", "news"
    post_id: str         # unique within source
    ticker: str          # target stock ticker
    title: str
    body: str
    author: str
    upvotes: int
    created_at: datetime
    language: str        # "ko" | "en"
    url: Optional[str] = None


@dataclass(frozen=True)
class NewsHeadline:
    """Immutable container for a news headline."""

    source: str
    headline_id: str
    ticker: str
    title: str
    summary: str
    url: str
    published_at: datetime
    language: str = "en"


@dataclass(frozen=True)
class FearGreedData:
    """CNN Fear & Greed index snapshot."""

    value: float           # 0–100
    label: str             # "Extreme Fear" … "Extreme Greed"
    collected_at: datetime
    one_week_ago: Optional[float] = None
    one_month_ago: Optional[float] = None


class BaseCrawler(ABC):
    """Abstract base for all crawlers."""

    @abstractmethod
    def crawl(self, ticker: str, **kwargs) -> tuple[RawPost, ...]:
        """Collect posts for *ticker* and return an immutable tuple."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this crawler can run (credentials present, etc.)."""
        ...
