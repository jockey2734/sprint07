"""Fundamental feature engineering: ratio changes, earnings surprise, YoY."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from stock_ai.fundamentals.statement_fetcher import FinancialStatements
from stock_ai.storage.cache import FileCache


class FundamentalFeatureBuilder:
    """Builds time-series fundamental features from financial statements."""

    def __init__(self, cache: Optional[FileCache] = None) -> None:
        self._cache = cache or FileCache()

    def build(
        self,
        ticker: str,
        statements: FinancialStatements,
        price_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Build fundamental feature DataFrame aligned to price_index.

        Features: PER change, ROE, debt ratio, earnings surprise, revenue growth,
                  EPS growth, gross margin, event flags (earnings date).
        """
        cache_key = f"fund_features:{ticker}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        is_df = statements.income_statement
        bs_df = statements.balance_sheet

        if is_df.empty:
            logger.warning(f"Fundamental features: empty income statement for {ticker}")
            return pd.DataFrame(index=price_index)

        # Build quarterly feature series
        quarterly_records: list[dict] = []
        for date in is_df.index:
            row = {"Date": date}
            is_row = is_df.loc[date] if date in is_df.index else pd.Series(dtype=float)
            bs_row = bs_df.loc[date] if (not bs_df.empty and date in bs_df.index) else pd.Series(dtype=float)

            row["net_income"] = self._get(is_row, ["Net Income", "NetIncome"])
            row["total_revenue"] = self._get(is_row, ["Total Revenue", "Revenue"])
            row["gross_profit"] = self._get(is_row, ["Gross Profit", "GrossProfit"])
            row["total_equity"] = self._get(bs_row, ["Total Stockholder Equity", "StockholdersEquity"])
            row["total_assets"] = self._get(bs_row, ["Total Assets", "TotalAssets"])
            row["total_liab"] = self._get(bs_row, ["Total Liab", "TotalLiabilitiesNetMinorityInterest"])

            # Derived ratios
            row["roe"] = self._div(row["net_income"], row["total_equity"])
            row["debt_ratio"] = self._div(row["total_liab"], row["total_assets"])
            row["gross_margin"] = self._div(row["gross_profit"], row["total_revenue"])
            quarterly_records.append(row)

        q_df = pd.DataFrame(quarterly_records).set_index("Date").sort_index()

        # YoY growth rates (4 quarters back)
        for col in ["net_income", "total_revenue"]:
            q_df[f"{col}_yoy"] = q_df[col].pct_change(4)

        # Earnings surprise: actual vs analyst estimate
        q_df = self._add_earnings_surprise(ticker, q_df)

        # Forward-fill to daily price index
        daily = q_df.reindex(price_index, method="ffill")

        self._cache.set(cache_key, daily, ttl=86400)
        logger.info(f"Fundamental features: {daily.shape} for {ticker}")
        return daily

    def _add_earnings_surprise(self, ticker: str, q_df: pd.DataFrame) -> pd.DataFrame:
        """Add earnings surprise feature (actual EPS - estimate EPS) / |estimate|."""
        try:
            tk = yf.Ticker(ticker)
            earnings = tk.earnings_history
            if earnings is None or (hasattr(earnings, "empty") and earnings.empty):
                return q_df

            result = q_df.copy()
            surprise_series = pd.Series(dtype=float, name="earnings_surprise")
            for idx, row in earnings.iterrows():
                actual = row.get("epsActual", None)
                estimate = row.get("epsEstimate", None)
                if actual is not None and estimate is not None and estimate != 0:
                    surprise = (actual - estimate) / abs(estimate)
                    surprise_series[idx] = surprise
            if not surprise_series.empty:
                result["earnings_surprise"] = surprise_series
            return result
        except Exception as exc:
            logger.debug(f"Earnings surprise fetch failed: {exc}")
            return q_df

    def _get(self, row: pd.Series, candidates: list[str]) -> Optional[float]:
        for name in candidates:
            if name in row.index and pd.notna(row[name]):
                return float(row[name])
        return None

    def _div(self, a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is not None and b is not None and b != 0:
            return a / b
        return None
