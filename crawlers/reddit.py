"""Reddit PRAW crawler — optional, requires .env credentials."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from loguru import logger

from stock_ai.crawlers.base import BaseCrawler, RawPost
from stock_ai.storage.cache import FileCache


def _has_reddit_credentials() -> bool:
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    return bool(
        client_id and client_secret
        and client_id != "your_client_id_here"
        and client_secret != "your_client_secret_here"
    )


class RedditCrawler(BaseCrawler):
    """Crawls Reddit posts from finance-related subreddits using PRAW."""

    def __init__(
        self,
        subreddits: Optional[list[str]] = None,
        limit: int = 100,
        cache: Optional[FileCache] = None,
    ) -> None:
        self._subreddits = subreddits or ["stocks", "investing", "wallstreetbets"]
        self._limit = limit
        self._cache = cache or FileCache()
        self._reddit = None  # lazy init

    def is_available(self) -> bool:
        return _has_reddit_credentials()

    def _get_reddit(self):
        if self._reddit is not None:
            return self._reddit
        try:
            import praw
            self._reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent=os.getenv("REDDIT_USER_AGENT", "StockAI/1.0"),
                read_only=True,
            )
            logger.info("Reddit client initialized")
            return self._reddit
        except Exception as exc:
            logger.error(f"Reddit init failed: {exc}")
            return None

    def crawl(self, ticker: str, **kwargs) -> tuple[RawPost, ...]:
        if not self.is_available():
            logger.warning("Reddit credentials not found — skipping Reddit crawl")
            return ()

        cache_key = f"reddit:{ticker}:{','.join(self._subreddits)}:{self._limit}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        reddit = self._get_reddit()
        if reddit is None:
            return ()

        # Clean ticker for search (remove exchange suffix)
        search_query = ticker.split(".")[0]
        posts: list[RawPost] = []

        for sub_name in self._subreddits:
            batch = self._fetch_subreddit(reddit, sub_name, search_query, ticker)
            posts.extend(batch)

        result = tuple(posts)
        self._cache.set(cache_key, result, ttl=3600)
        logger.info(f"Reddit: crawled {len(result)} posts for {ticker}")
        return result

    def _fetch_subreddit(self, reddit, sub_name: str, query: str, ticker: str) -> list[RawPost]:
        posts: list[RawPost] = []
        try:
            sub = reddit.subreddit(sub_name)
            for submission in sub.search(query, limit=self._limit // len(self._subreddits), sort="new"):
                created = datetime.utcfromtimestamp(submission.created_utc)
                posts.append(
                    RawPost(
                        source="reddit",
                        post_id=submission.id,
                        ticker=ticker,
                        title=submission.title,
                        body=submission.selftext[:2000],  # truncate long posts
                        author=str(submission.author) if submission.author else "[deleted]",
                        upvotes=submission.score,
                        created_at=created,
                        language="en",
                        url=f"https://www.reddit.com{submission.permalink}",
                    )
                )
        except Exception as exc:
            logger.warning(f"Reddit subreddit {sub_name} error: {exc}")
        return posts
