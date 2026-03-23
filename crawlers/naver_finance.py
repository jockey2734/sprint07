"""Naver Finance discussion board crawler (requests + BeautifulSoup4)."""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup
from loguru import logger

from stock_ai.crawlers.base import BaseCrawler, RawPost
from stock_ai.storage.cache import FileCache

_BOARD_URL = "https://finance.naver.com/item/board.naver"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}


def _parse_naver_date(text: str) -> datetime:
    """Parse Naver date strings like '2024.01.15 09:30' or '01.15 09:30'."""
    text = text.strip()
    try:
        if len(text) == 16 and text[4] == ".":
            return datetime.strptime(text, "%Y.%m.%d %H:%M")
        if len(text) == 11:
            year = datetime.now().year
            return datetime.strptime(f"{year}.{text}", "%Y.%m.%d %H:%M")
    except ValueError:
        pass
    return datetime.now()


def _ticker_to_code(ticker: str) -> str:
    """Strip market suffix and return numeric code (e.g. '005930.KS' → '005930')."""
    return re.split(r"\.", ticker)[0]


class NaverFinanceCrawler(BaseCrawler):
    """Crawls Naver Finance 종목토론실 discussion board."""

    def __init__(self, cache: Optional[FileCache] = None, max_pages: int = 5) -> None:
        self._cache = cache or FileCache()
        self._max_pages = max_pages

    def is_available(self) -> bool:
        return True  # No credentials required

    def crawl(self, ticker: str, **kwargs) -> tuple[RawPost, ...]:
        code = _ticker_to_code(ticker)
        if not code.isdigit():
            logger.warning(f"NaverFinanceCrawler: {ticker} is not a KRX ticker, skipping")
            return ()

        cache_key = f"naver:{ticker}:{self._max_pages}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        posts: list[RawPost] = []
        for page in range(1, self._max_pages + 1):
            batch = self._fetch_page(ticker, code, page)
            posts.extend(batch)
            if len(batch) == 0:
                break
            time.sleep(0.5)  # polite delay

        result = tuple(posts)
        self._cache.set(cache_key, result, ttl=3600)
        logger.info(f"NaverFinance: crawled {len(result)} posts for {ticker}")
        return result

    def _fetch_page(self, ticker: str, code: str, page: int) -> list[RawPost]:
        params = urlencode({"code": code, "page": page})
        url = f"{_BOARD_URL}?{params}"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"NaverFinance fetch failed (page {page}): {exc}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.select("table.type2 tr")
        posts: list[RawPost] = []

        for row in rows:
            cols = row.select("td")
            if len(cols) < 5:
                continue
            try:
                title_tag = cols[1].select_one("a")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                href = title_tag.get("href", "")
                post_id = re.search(r"nid=(\d+)", href)
                if not post_id:
                    continue

                date_text = cols[3].get_text(strip=True)
                upvote_text = cols[4].get_text(strip=True).replace(",", "")
                upvotes = int(upvote_text) if upvote_text.isdigit() else 0
                author = cols[2].get_text(strip=True)

                posts.append(
                    RawPost(
                        source="naver",
                        post_id=post_id.group(1),
                        ticker=ticker,
                        title=title,
                        body="",  # Body requires a separate request
                        author=author,
                        upvotes=upvotes,
                        created_at=_parse_naver_date(date_text),
                        language="ko",
                        url=f"https://finance.naver.com{href}",
                    )
                )
            except Exception as exc:
                logger.debug(f"Row parse error: {exc}")
                continue

        return posts
