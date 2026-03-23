"""News crawler: Google News RSS + Naver News + financial news RSS."""

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

from stock_ai.crawlers.base import BaseCrawler, NewsHeadline, RawPost
from stock_ai.storage.cache import FileCache

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _entry_to_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _parse_feed_date(entry) -> datetime:
    for field in ("published", "updated"):
        val = getattr(entry, field, None)
        if val:
            try:
                return parsedate_to_datetime(val).replace(tzinfo=None)
            except Exception:
                pass
    return datetime.utcnow()


class NewsCrawler(BaseCrawler):
    """Aggregates news from Google News RSS and Naver News."""

    def __init__(self, cache: Optional[FileCache] = None, max_articles: int = 50) -> None:
        self._cache = cache or FileCache()
        self._max_articles = max_articles

    def is_available(self) -> bool:
        return True

    def crawl(self, ticker: str, **kwargs) -> tuple[RawPost, ...]:
        """Return news as RawPost objects for compatibility with sentiment pipeline."""
        headlines = self.crawl_headlines(ticker)
        posts = tuple(
            RawPost(
                source=h.source,
                post_id=h.headline_id,
                ticker=h.ticker,
                title=h.title,
                body=h.summary,
                author="",
                upvotes=0,
                created_at=h.published_at,
                language=h.language,
                url=h.url,
            )
            for h in headlines
        )
        return posts

    def crawl_headlines(self, ticker: str) -> tuple[NewsHeadline, ...]:
        cache_key = f"news:{ticker}:{self._max_articles}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        symbol = ticker.split(".")[0]
        is_korean = ticker.endswith(".KS") or ticker.endswith(".KQ")

        headlines: list[NewsHeadline] = []
        headlines.extend(self._google_news_rss(symbol, ticker, is_korean))
        if is_korean:
            headlines.extend(self._naver_news(symbol, ticker))

        # Deduplicate by url
        seen: set[str] = set()
        unique: list[NewsHeadline] = []
        for h in headlines:
            if h.url not in seen:
                seen.add(h.url)
                unique.append(h)

        result = tuple(unique[: self._max_articles])
        self._cache.set(cache_key, result, ttl=3600)
        logger.info(f"News: collected {len(result)} headlines for {ticker}")
        return result

    def _google_news_rss(self, symbol: str, ticker: str, is_korean: bool) -> list[NewsHeadline]:
        lang = "ko" if is_korean else "en"
        hl = "ko" if is_korean else "en-US"
        gl = "KR" if is_korean else "US"
        ceid = "KR:ko" if is_korean else "US:en"
        query = quote(f"{symbol} stock" if not is_korean else f"{symbol} 주식")
        url = f"https://news.google.com/rss/search?q={query}&hl={hl}&gl={gl}&ceid={ceid}"

        results: list[NewsHeadline] = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:25]:
                link = getattr(entry, "link", "") or ""
                results.append(
                    NewsHeadline(
                        source="google_news",
                        headline_id=_entry_to_id(link),
                        ticker=ticker,
                        title=getattr(entry, "title", ""),
                        summary=getattr(entry, "summary", ""),
                        url=link,
                        published_at=_parse_feed_date(entry),
                        language=lang,
                    )
                )
        except Exception as exc:
            logger.warning(f"Google News RSS failed for {ticker}: {exc}")
        return results

    def _naver_news(self, symbol: str, ticker: str) -> list[NewsHeadline]:
        url = f"https://finance.naver.com/item/news_news.naver?code={symbol}&page=1"
        results: list[NewsHeadline] = []
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for row in soup.select("table.type5 tr"):
                cols = row.select("td")
                if len(cols) < 3:
                    continue
                a_tag = cols[0].select_one("a")
                if not a_tag:
                    continue
                title = a_tag.get_text(strip=True)
                href = a_tag.get("href", "")
                full_url = f"https://finance.naver.com{href}" if href.startswith("/") else href
                date_text = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                try:
                    pub_dt = datetime.strptime(date_text, "%Y.%m.%d %H:%M")
                except ValueError:
                    pub_dt = datetime.utcnow()

                results.append(
                    NewsHeadline(
                        source="naver_news",
                        headline_id=_entry_to_id(full_url),
                        ticker=ticker,
                        title=title,
                        summary="",
                        url=full_url,
                        published_at=pub_dt,
                        language="ko",
                    )
                )
        except Exception as exc:
            logger.warning(f"Naver news failed for {ticker}: {exc}")
        return results
