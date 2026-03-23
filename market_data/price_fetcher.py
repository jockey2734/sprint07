"""yfinance OHLCV price fetcher with file cache."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from stock_ai.storage.cache import FileCache


class PriceFetcher:
    """Fetches OHLCV price data from yfinance with TTL cache."""

    def __init__(self, cache: Optional[FileCache] = None, ttl: int = 86400) -> None:
        self._cache = cache or FileCache()
        self._ttl = ttl

    def fetch(
        self,
        ticker: str,
        start: str = "2020-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame indexed by date.

        Args:
            ticker: Stock ticker (e.g. "AAPL", "005930.KS").
            start: Start date string "YYYY-MM-DD".
            end: End date string (default = today).

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume, Adj Close].
        """
        end_str = end or date.today().isoformat()
        cache_key = f"price:{ticker}:{start}:{end_str}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Price cache hit: {ticker}")
            return cached

        try:
            tk = yf.Ticker(ticker)
            df = tk.history(start=start, end=end_str, auto_adjust=True)
            if df.empty:
                logger.warning(f"No price data for {ticker} ({start}→{end_str})")
                return pd.DataFrame()

            # Normalize column names
            df = df.rename(columns=str.title)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"

            self._cache.set(cache_key, df, ttl=self._ttl)
            logger.info(f"Price fetched: {ticker} — {len(df)} rows")
            return df
        except Exception as exc:
            logger.error(f"Price fetch failed for {ticker}: {exc}")
            return pd.DataFrame()

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Return the latest closing price."""
        df = self.fetch(ticker)
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])

    def get_company_info(self, ticker: str) -> dict:
        """Return basic company info dict."""
        try:
            tk = yf.Ticker(ticker)
            return tk.info or {}
        except Exception as exc:
            logger.warning(f"Company info failed for {ticker}: {exc}")
            return {}
