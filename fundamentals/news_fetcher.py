"""Earnings / disclosure news fetcher (Seeking Alpha RSS + Naver Finance)."""

from __future__ import annotations

import hashlib
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote

import feedparser
import requests
from bs4 import BeautifulSoup
from loguru import logger

from stock_ai.crawlers.base import NewsHeadline
from stock_ai.storage.cache import FileCache

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _entry_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _parse_date(entry) -> datetime:
    for field in ("published", "updated"):
        val = getattr(entry, field, None)
        if val:
            try:
                return parsedate_to_datetime(val).replace(tzinfo=None)
            except Exception:
                pass
    return datetime.utcnow()


class FundamentalsNewsFetcher:
    """Fetches earnings/disclosure news for fundamental analysis."""

    def __init__(self, cache: Optional[FileCache] = None, ttl: int = 3600) -> None:
        self._cache = cache or FileCache()
        self._ttl = ttl

    def fetch(self, ticker: str) -> tuple[NewsHeadline, ...]:
        """Fetch earnings/disclosure news. Returns immutable tuple."""
        cache_key = f"fund_news:{ticker}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        symbol = ticker.split(".")[0]
        is_korean = ticker.endswith(".KS") or ticker.endswith(".KQ")

        headlines: list[NewsHeadline] = []
        headlines.extend(self._seeking_alpha_rss(symbol, ticker))
        if is_korean:
            headlines.extend(self._naver_dart_news(symbol, ticker))

        # Deduplicate
        seen: set[str] = set()
        unique: list[NewsHeadline] = []
        for h in headlines:
            if h.url not in seen:
                seen.add(h.url)
                unique.append(h)

        result = tuple(unique)
        self._cache.set(cache_key, result, ttl=self._ttl)
        logger.info(f"FundamentalsNews: {len(result)} articles for {ticker}")
        return result

    def _seeking_alpha_rss(self, symbol: str, ticker: str) -> list[NewsHeadline]:
        """Seeking Alpha RSS feed for US stocks."""
        url = f"https://seekingalpha.com/api/sa/combined/{symbol}.xml"
        results: list[NewsHeadline] = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                link = getattr(entry, "link", "") or ""
                results.append(
                    NewsHeadline(
                        source="seeking_alpha",
                        headline_id=_entry_id(link),
                        ticker=ticker,
                        title=getattr(entry, "title", ""),
                        summary=getattr(entry, "summary", "")[:500],
                        url=link,
                        published_at=_parse_date(entry),
                        language="en",
                    )
                )
        except Exception as exc:
            logger.debug(f"Seeking Alpha RSS failed for {ticker}: {exc}")
        return results

    def _naver_dart_news(self, symbol: str, ticker: str) -> list[NewsHeadline]:
        """Naver Finance disclosure news for Korean stocks."""
        url = f"https://finance.naver.com/item/news_notice.naver?code={symbol}&page=1"
        results: list[NewsHeadline] = []
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for row in soup.select("table.type5 tr"):
                cols = row.select("td")
                if len(cols) < 2:
                    continue
                a_tag = cols[0].select_one("a")
                if not a_tag:
                    continue
                title = a_tag.get_text(strip=True)
                href = a_tag.get("href", "")
                full_url = f"https://finance.naver.com{href}" if href.startswith("/") else href
                date_text = cols[-1].get_text(strip=True) if cols else ""
                try:
                    pub_dt = datetime.strptime(date_text, "%Y.%m.%d")
                except ValueError:
                    pub_dt = datetime.utcnow()
                results.append(
                    NewsHeadline(
                        source="naver_dart",
                        headline_id=_entry_id(full_url),
                        ticker=ticker,
                        title=title,
                        summary="",
                        url=full_url,
                        published_at=pub_dt,
                        language="ko",
                    )
                )
        except Exception as exc:
            logger.debug(f"Naver DART news failed for {ticker}: {exc}")
        return results
