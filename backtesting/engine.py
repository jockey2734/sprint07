"""Walk-forward backtesting engine with 5-fold time-series cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from stock_ai.backtesting.metrics import BacktestMetrics, compute_metrics
from stock_ai.backtesting.strategy import BaseStrategy, Signal, TradeSignal


@dataclass(frozen=True)
class FoldResult:
    """Results for a single walk-forward fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: BacktestMetrics
    equity_curve: pd.Series
    trade_returns: pd.Series


@dataclass(frozen=True)
class BacktestResult:
    """Aggregated walk-forward backtest results."""

    ticker: str
    strategy_name: str
    folds: tuple[FoldResult, ...]
    avg_sharpe: float
    avg_max_drawdown: float
    avg_win_rate: float
    avg_total_return: float
    total_trades: int


class BacktestEngine:
    """Walk-forward backtesting engine."""

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
    ) -> None:
        self._n_splits = n_splits
        self._train_ratio = train_ratio
        self._initial_capital = initial_capital
        self._commission = commission

    def run_walk_forward(
        self,
        price_df: pd.DataFrame,
        strategy: BaseStrategy,
        predictions_df: Optional[pd.DataFrame] = None,
        ticker: str = "unknown",
    ) -> BacktestResult:
        """Execute walk-forward backtest.

        Args:
            price_df: OHLCV price DataFrame indexed by date.
            strategy: Trading strategy instance.
            predictions_df: Optional model predictions indexed by date.
            ticker: Ticker symbol for labeling.

        Returns:
            BacktestResult with per-fold and aggregate metrics.
        """
        if price_df.empty or len(price_df) < 50:
            logger.warning(f"BacktestEngine: insufficient data ({len(price_df)} rows)")
            return self._empty_result(ticker, strategy.__class__.__name__)

        n = len(price_df)
        fold_size = n // (self._n_splits + 1)
        fold_results: list[FoldResult] = []

        for split in range(self._n_splits):
            train_end_idx = fold_size * (split + 1)
            test_end_idx = min(train_end_idx + fold_size, n)

            if test_end_idx <= train_end_idx:
                break

            train_slice = price_df.iloc[:train_end_idx]
            test_slice = price_df.iloc[train_end_idx:test_end_idx]

            pred_slice = None
            if predictions_df is not None and not predictions_df.empty:
                pred_slice = predictions_df[
                    predictions_df.index.isin(test_slice.index)
                ]

            signals = strategy.generate_signals(test_slice, pred_slice)
            equity, trade_rets = self._simulate(test_slice, signals)
            metrics = compute_metrics(equity, trade_rets)

            fold_results.append(
                FoldResult(
                    fold=split + 1,
                    train_start=train_slice.index[0],
                    train_end=train_slice.index[-1],
                    test_start=test_slice.index[0],
                    test_end=test_slice.index[-1],
                    metrics=metrics,
                    equity_curve=equity,
                    trade_returns=trade_rets,
                )
            )

        if not fold_results:
            return self._empty_result(ticker, strategy.__class__.__name__)

        avg_sharpe = float(np.mean([f.metrics.sharpe_ratio for f in fold_results]))
        avg_mdd = float(np.mean([f.metrics.max_drawdown for f in fold_results]))
        avg_wr = float(np.mean([f.metrics.win_rate for f in fold_results]))
        avg_ret = float(np.mean([f.metrics.total_return for f in fold_results]))
        total_trades = sum(f.metrics.n_trades for f in fold_results)

        logger.info(
            f"Backtest {ticker} ({strategy.__class__.__name__}): "
            f"Sharpe={avg_sharpe:.2f}, MDD={avg_mdd:.1%}, WinRate={avg_wr:.1%}"
        )

        return BacktestResult(
            ticker=ticker,
            strategy_name=strategy.__class__.__name__,
            folds=tuple(fold_results),
            avg_sharpe=avg_sharpe,
            avg_max_drawdown=avg_mdd,
            avg_win_rate=avg_wr,
            avg_total_return=avg_ret,
            total_trades=total_trades,
        )

    def _simulate(
        self,
        price_df: pd.DataFrame,
        signals: tuple[TradeSignal, ...],
    ) -> tuple[pd.Series, pd.Series]:
        """Simulate portfolio equity and return per-trade series."""
        signal_map = {s.date: s for s in signals}
        capital = self._initial_capital
        position = 0.0
        entry_price = 0.0
        equity_list: list[tuple] = []
        trade_returns: list[float] = []

        for date, row in price_df.iterrows():
            close = float(row.get("Close", 0))
            signal = signal_map.get(date, None)

            # Execute trade signal
            if signal is not None:
                if signal.signal == Signal.BUY and position == 0 and close > 0:
                    # Enter long
                    shares = (capital * (1 - self._commission)) / close
                    position = shares
                    entry_price = close
                    capital = 0.0

                elif signal.signal == Signal.SELL and position > 0 and close > 0:
                    # Exit long
                    proceeds = position * close * (1 - self._commission)
                    trade_ret = (proceeds / (position * entry_price)) - 1
                    trade_returns.append(trade_ret)
                    capital = proceeds
                    position = 0.0
                    entry_price = 0.0

            # Mark-to-market equity
            current_value = capital + (position * close if position > 0 else 0)
            equity_list.append((date, current_value))

        # Close any open position at last price
        if position > 0 and not price_df.empty:
            last_close = float(price_df["Close"].iloc[-1])
            proceeds = position * last_close * (1 - self._commission)
            trade_ret = (proceeds / (position * entry_price)) - 1
            trade_returns.append(trade_ret)

        equity = pd.Series(
            [v for _, v in equity_list],
            index=[d for d, _ in equity_list],
            name="equity",
        )
        trade_series = pd.Series(trade_returns, name="trade_returns")
        return equity, trade_series

    def _empty_result(self, ticker: str, strategy_name: str) -> BacktestResult:
        from stock_ai.backtesting.metrics import _zero_metrics
        return BacktestResult(
            ticker=ticker,
            strategy_name=strategy_name,
            folds=(),
            avg_sharpe=0.0,
            avg_max_drawdown=0.0,
            avg_win_rate=0.0,
            avg_total_return=0.0,
            total_trades=0,
        )
