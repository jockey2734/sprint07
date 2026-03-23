"""Technical indicator computation using the `ta` library."""

from __future__ import annotations

import pandas as pd
from loguru import logger

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    logger.warning("ta library not installed — technical indicators disabled")


class TechnicalIndicators:
    """Computes RSI, MACD, Bollinger Bands, SMA, EMA, ATR."""

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: int = 2,
        sma_periods: list[int] | None = None,
        ema_periods: list[int] | None = None,
        atr_period: int = 14,
    ) -> None:
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sma_periods = sma_periods or [5, 10, 20, 50, 200]
        self.ema_periods = ema_periods or [12, 26]
        self.atr_period = atr_period

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and return enriched DataFrame.

        Requires columns: Close, High, Low, Volume.
        """
        if df.empty:
            return df

        if not _TA_AVAILABLE:
            return self._compute_manual(df)

        result = df.copy()
        close = result["Close"]
        high = result.get("High", close)
        low = result.get("Low", close)
        volume = result.get("Volume", pd.Series(0, index=result.index))

        # RSI
        result["rsi"] = ta.momentum.RSIIndicator(close, window=self.rsi_period).rsi()

        # MACD
        macd = ta.trend.MACD(close, window_fast=self.macd_fast,
                             window_slow=self.macd_slow, window_sign=self.macd_signal)
        result["macd"] = macd.macd()
        result["macd_signal"] = macd.macd_signal()
        result["macd_diff"] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=self.bb_period, window_dev=self.bb_std)
        result["bb_upper"] = bb.bollinger_hband()
        result["bb_lower"] = bb.bollinger_lband()
        result["bb_middle"] = bb.bollinger_mavg()
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]
        result["bb_pct"] = bb.bollinger_pband()

        # SMAs
        for period in self.sma_periods:
            if len(result) >= period:
                result[f"sma_{period}"] = ta.trend.SMAIndicator(close, window=period).sma_indicator()

        # EMAs
        for period in self.ema_periods:
            if len(result) >= period:
                result[f"ema_{period}"] = ta.trend.EMAIndicator(close, window=period).ema_indicator()

        # ATR
        if "High" in df.columns and "Low" in df.columns:
            result["atr"] = ta.volatility.AverageTrueRange(
                high, low, close, window=self.atr_period
            ).average_true_range()

        # OBV (volume)
        if "Volume" in df.columns:
            result["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        logger.debug(f"Technical indicators computed: {len(result.columns)} columns")
        return result

    def _compute_manual(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback manual computation if ta library unavailable."""
        result = df.copy()
        close = result["Close"]

        # SMA only
        for period in self.sma_periods:
            if len(result) >= period:
                result[f"sma_{period}"] = close.rolling(period).mean()

        # Simple RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-9)
        result["rsi"] = 100 - (100 / (1 + rs))

        return result
