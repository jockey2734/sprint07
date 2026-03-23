"""Financial ratio computation: PER, PBR, ROE, EPS growth, debt ratio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from stock_ai.fundamentals.statement_fetcher import FinancialStatements
from stock_ai.storage.cache import FileCache


@dataclass(frozen=True)
class FundamentalRatios:
    """Point-in-time financial ratios."""

    ticker: str
    per: Optional[float]
    pbr: Optional[float]
    roe: Optional[float]
    debt_ratio: Optional[float]
    eps: Optional[float]
    eps_growth_yoy: Optional[float]   # year-over-year
    revenue_growth_yoy: Optional[float]
    current_ratio: Optional[float]
    gross_margin: Optional[float]


class RatiosCalculator:
    """Computes financial ratios from financial statements + live price."""

    def __init__(self, cache: Optional[FileCache] = None) -> None:
        self._cache = cache or FileCache()

    def compute_latest(self, ticker: str, statements: FinancialStatements) -> FundamentalRatios:
        """Compute current fundamental ratios for ticker."""
        cache_key = f"ratios_latest:{ticker}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}

            per = self._safe_float(info.get("trailingPE"))
            pbr = self._safe_float(info.get("priceToBook"))
            roe = self._safe_float(info.get("returnOnEquity"))
            eps = self._safe_float(info.get("trailingEps"))
            current_ratio = self._safe_float(info.get("currentRatio"))
            gross_margin = self._safe_float(info.get("grossMargins"))

            debt_ratio = self._compute_debt_ratio(statements)
            eps_growth = self._compute_eps_growth_yoy(statements)
            rev_growth = self._compute_revenue_growth_yoy(statements)

            result = FundamentalRatios(
                ticker=ticker,
                per=per,
                pbr=pbr,
                roe=roe,
                debt_ratio=debt_ratio,
                eps=eps,
                eps_growth_yoy=eps_growth,
                revenue_growth_yoy=rev_growth,
                current_ratio=current_ratio,
                gross_margin=gross_margin,
            )
            self._cache.set(cache_key, result, ttl=86400)
            logger.info(f"Ratios: {ticker} — PER={per}, PBR={pbr}, ROE={roe:.3f}" if roe else f"Ratios: {ticker} computed")
            return result
        except Exception as exc:
            logger.error(f"Ratio computation failed for {ticker}: {exc}")
            return FundamentalRatios(ticker=ticker, per=None, pbr=None, roe=None,
                                     debt_ratio=None, eps=None, eps_growth_yoy=None,
                                     revenue_growth_yoy=None, current_ratio=None,
                                     gross_margin=None)

    def compute_time_series(self, statements: FinancialStatements) -> pd.DataFrame:
        """Build quarterly ratio time series from financial statements."""
        is_df = statements.income_statement
        bs_df = statements.balance_sheet
        if is_df.empty or bs_df.empty:
            return pd.DataFrame()

        records: list[dict] = []
        for date in is_df.index:
            row: dict = {"Date": date}
            is_row = is_df.loc[date] if date in is_df.index else pd.Series(dtype=float)
            bs_row = bs_df.loc[date] if date in bs_df.index else pd.Series(dtype=float)

            net_income = self._get_field(is_row, ["Net Income", "NetIncome"])
            total_equity = self._get_field(bs_row, ["Total Stockholder Equity", "StockholdersEquity", "Total Equity"])
            total_assets = self._get_field(bs_row, ["Total Assets", "TotalAssets"])
            total_liab = self._get_field(bs_row, ["Total Liab", "TotalLiabilitiesNetMinorityInterest"])

            row["net_income"] = net_income
            row["total_equity"] = total_equity
            row["total_assets"] = total_assets
            row["total_liabilities"] = total_liab
            row["roe"] = (net_income / total_equity) if (total_equity and total_equity != 0) else None
            row["debt_ratio"] = (total_liab / total_assets) if (total_assets and total_assets != 0) else None
            records.append(row)

        df = pd.DataFrame(records).set_index("Date").sort_index()
        return df

    def _compute_debt_ratio(self, statements: FinancialStatements) -> Optional[float]:
        bs = statements.balance_sheet
        if bs.empty:
            return None
        latest = bs.iloc[-1]
        assets = self._get_field(latest, ["Total Assets", "TotalAssets"])
        liab = self._get_field(latest, ["Total Liab", "TotalLiabilitiesNetMinorityInterest"])
        if assets and liab and assets != 0:
            return liab / assets
        return None

    def _compute_eps_growth_yoy(self, statements: FinancialStatements) -> Optional[float]:
        is_df = statements.income_statement
        if is_df.empty or len(is_df) < 5:
            return None
        try:
            latest = self._get_field(is_df.iloc[-1], ["Diluted EPS", "Basic EPS", "EPS"])
            year_ago = self._get_field(is_df.iloc[-5], ["Diluted EPS", "Basic EPS", "EPS"])
            if latest is not None and year_ago and year_ago != 0:
                return (latest - year_ago) / abs(year_ago)
        except Exception:
            pass
        return None

    def _compute_revenue_growth_yoy(self, statements: FinancialStatements) -> Optional[float]:
        is_df = statements.income_statement
        if is_df.empty or len(is_df) < 5:
            return None
        try:
            latest = self._get_field(is_df.iloc[-1], ["Total Revenue", "Revenue"])
            year_ago = self._get_field(is_df.iloc[-5], ["Total Revenue", "Revenue"])
            if latest is not None and year_ago and year_ago != 0:
                return (latest - year_ago) / abs(year_ago)
        except Exception:
            pass
        return None

    def _get_field(self, row: pd.Series, candidates: list[str]) -> Optional[float]:
        for name in candidates:
            if name in row.index and pd.notna(row[name]):
                return float(row[name])
        return None

    def _safe_float(self, value) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
