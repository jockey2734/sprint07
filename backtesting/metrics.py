"""Backtesting performance metrics: Sharpe, MDD, Sortino, win rate, profit factor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    """Immutable set of performance metrics for a backtested strategy."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float          # negative fraction (e.g. -0.25 = -25%)
    win_rate: float              # fraction of winning trades
    profit_factor: float         # gross profit / gross loss
    n_trades: int
    avg_trade_return: float
    volatility: float            # annualized


def compute_metrics(
    equity_curve: pd.Series,
    trade_returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """Compute comprehensive performance metrics.

    Args:
        equity_curve: Portfolio value over time (indexed by date).
        trade_returns: Per-trade return fractions.
        risk_free_rate: Annual risk-free rate (default 4%).
        periods_per_year: Trading days per year.

    Returns:
        Immutable BacktestMetrics instance.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return _zero_metrics()

    daily_returns = equity_curve.pct_change().dropna()

    total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0)
    n_periods = len(daily_returns)
    n_years = n_periods / periods_per_year
    annualized_return = float((1 + total_return) ** (1.0 / max(n_years, 1e-9)) - 1)

    volatility = float(daily_returns.std() * np.sqrt(periods_per_year))

    rf_daily = risk_free_rate / periods_per_year
    excess = daily_returns - rf_daily
    sharpe_raw = float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(periods_per_year))
    sharpe = float(np.clip(sharpe_raw, -100.0, 100.0))

    # Sortino: only downside deviation
    downside = daily_returns[daily_returns < 0]
    sortino_denom = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) > 0 else 1e-9
    sortino = float((annualized_return - risk_free_rate) / (sortino_denom + 1e-9))

    # Max Drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # Trade-level metrics
    n_trades = len(trade_returns)
    if n_trades > 0:
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns <= 0]
        win_rate = float(len(wins) / n_trades)
        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = float(losses.abs().sum()) if not losses.empty else 1e-9
        profit_factor = gross_profit / gross_loss
        avg_trade = float(trade_returns.mean())
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=n_trades,
        avg_trade_return=avg_trade,
        volatility=volatility,
    )


def _zero_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        total_return=0.0,
        annualized_return=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        n_trades=0,
        avg_trade_return=0.0,
        volatility=0.0,
    )
