"""Trading strategies for backtesting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class TradeSignal:
    """Immutable trade signal at a specific date."""

    date: pd.Timestamp
    signal: Signal
    strength: float  # 0–1, signal conviction
    reason: str


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    @abstractmethod
    def generate_signals(
        self, price_df: pd.DataFrame, predictions: Optional[pd.DataFrame] = None
    ) -> tuple[TradeSignal, ...]:
        """Generate trade signals from price data and optional model predictions."""
        ...


class ThresholdStrategy(BaseStrategy):
    """Threshold-based strategy: buy if predicted return > threshold, sell if < -threshold."""

    def __init__(self, threshold_buy: float = 0.02, threshold_sell: float = -0.01) -> None:
        self._threshold_buy = threshold_buy
        self._threshold_sell = threshold_sell

    def generate_signals(
        self, price_df: pd.DataFrame, predictions: Optional[pd.DataFrame] = None
    ) -> tuple[TradeSignal, ...]:
        if predictions is None or predictions.empty:
            return ()

        signals: list[TradeSignal] = []
        for date, row in predictions.iterrows():
            ret = row.get("predicted_return", 0.0)
            if ret >= self._threshold_buy:
                signals.append(
                    TradeSignal(
                        date=date,
                        signal=Signal.BUY,
                        strength=min(1.0, ret / self._threshold_buy),
                        reason=f"predicted_return={ret:.3f} > threshold={self._threshold_buy}",
                    )
                )
            elif ret <= self._threshold_sell:
                signals.append(
                    TradeSignal(
                        date=date,
                        signal=Signal.SELL,
                        strength=min(1.0, abs(ret) / abs(self._threshold_sell)),
                        reason=f"predicted_return={ret:.3f} < threshold={self._threshold_sell}",
                    )
                )
        return tuple(signals)


class SentimentMomentumStrategy(BaseStrategy):
    """Combined sentiment momentum + price trend strategy."""

    def __init__(
        self,
        sentiment_threshold: float = 0.2,
        momentum_window: int = 5,
    ) -> None:
        self._sent_threshold = sentiment_threshold
        self._momentum_window = momentum_window

    def generate_signals(
        self, price_df: pd.DataFrame, predictions: Optional[pd.DataFrame] = None
    ) -> tuple[TradeSignal, ...]:
        signals: list[TradeSignal] = []

        if "compound" not in price_df.columns or "Close" not in price_df.columns:
            return ()

        df = price_df.copy()
        df["price_momentum"] = df["Close"].pct_change(self._momentum_window)

        for date, row in df.iterrows():
            sentiment = row.get("compound", 0.0)
            momentum = row.get("price_momentum", 0.0)

            if sentiment > self._sent_threshold and momentum > 0:
                signals.append(
                    TradeSignal(
                        date=date,
                        signal=Signal.BUY,
                        strength=min(1.0, sentiment),
                        reason=f"sentiment={sentiment:.2f} + momentum={momentum:.3f}",
                    )
                )
            elif sentiment < -self._sent_threshold and momentum < 0:
                signals.append(
                    TradeSignal(
                        date=date,
                        signal=Signal.SELL,
                        strength=min(1.0, abs(sentiment)),
                        reason=f"sentiment={sentiment:.2f} + momentum={momentum:.3f}",
                    )
                )

        return tuple(signals)
