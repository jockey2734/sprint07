"""Quarterly financial statement fetcher via yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from stock_ai.storage.cache import FileCache


@dataclass(frozen=True)
class FinancialStatements:
    """Container for the three core financial statements."""

    income_statement: pd.DataFrame   # quarterly
    balance_sheet: pd.DataFrame      # quarterly
    cash_flow: pd.DataFrame          # quarterly
    ticker: str


class StatementFetcher:
    """Fetches quarterly financial statements using yfinance."""

    def __init__(self, cache: Optional[FileCache] = None, ttl: int = 86400) -> None:
        self._cache = cache or FileCache()
        self._ttl = ttl

    def fetch(self, ticker: str) -> FinancialStatements:
        """Fetch and return all three quarterly financial statements.

        Returns empty DataFrames on failure.
        """
        cache_key = f"financials:{ticker}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Financial statements cache hit: {ticker}")
            return cached

        try:
            tk = yf.Ticker(ticker)
            income = self._safe_fetch(tk.quarterly_financials, "income statement", ticker)
            balance = self._safe_fetch(tk.quarterly_balance_sheet, "balance sheet", ticker)
            cashflow = self._safe_fetch(tk.quarterly_cashflow, "cash flow", ticker)

            result = FinancialStatements(
                income_statement=income,
                balance_sheet=balance,
                cash_flow=cashflow,
                ticker=ticker,
            )
            self._cache.set(cache_key, result, ttl=self._ttl)
            logger.info(
                f"Financials fetched: {ticker} "
                f"(IS: {income.shape}, BS: {balance.shape}, CF: {cashflow.shape})"
            )
            return result
        except Exception as exc:
            logger.error(f"Financial statement fetch failed for {ticker}: {exc}")
            empty = pd.DataFrame()
            return FinancialStatements(
                income_statement=empty,
                balance_sheet=empty,
                cash_flow=empty,
                ticker=ticker,
            )

    def _safe_fetch(self, df_or_none, name: str, ticker: str) -> pd.DataFrame:
        try:
            if df_or_none is None or (hasattr(df_or_none, "empty") and df_or_none.empty):
                logger.warning(f"Empty {name} for {ticker}")
                return pd.DataFrame()
            df = df_or_none.T  # Transpose: rows=quarters, cols=fields
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            return df.sort_index()
        except Exception as exc:
            logger.warning(f"Failed to process {name} for {ticker}: {exc}")
            return pd.DataFrame()
