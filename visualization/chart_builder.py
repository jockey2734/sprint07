"""Plotly interactive charts: candlestick, prediction comparison, sentiment timeline."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from stock_ai.models.base_model import PredictionResult


class ChartBuilder:
    """Builds Plotly figures for the stock AI dashboard."""

    def candlestick_with_indicators(
        self,
        price_df: pd.DataFrame,
        ticker: str,
        show_volume: bool = True,
    ) -> go.Figure:
        """OHLCV candlestick chart with technical indicators overlay."""
        rows = 2 if show_volume else 1
        row_heights = [0.75, 0.25] if show_volume else [1.0]
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights)

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=price_df.index,
                open=price_df.get("Open", price_df["Close"]),
                high=price_df.get("High", price_df["Close"]),
                low=price_df.get("Low", price_df["Close"]),
                close=price_df["Close"],
                name=ticker,
            ),
            row=1, col=1,
        )

        # SMA overlays
        for col in price_df.columns:
            if col.startswith("sma_"):
                period = col.split("_")[1]
                fig.add_trace(
                    go.Scatter(x=price_df.index, y=price_df[col], name=f"SMA {period}", line={"width": 1}),
                    row=1, col=1,
                )

        # Bollinger Bands
        if "bb_upper" in price_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=price_df.index, y=price_df["bb_upper"], name="BB Upper",
                    line={"color": "rgba(150,150,150,0.5)", "dash": "dash"}, showlegend=False,
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=price_df.index, y=price_df["bb_lower"], name="BB Lower",
                    fill="tonexty", fillcolor="rgba(150,150,150,0.1)",
                    line={"color": "rgba(150,150,150,0.5)", "dash": "dash"},
                ),
                row=1, col=1,
            )

        # Volume
        if show_volume and "Volume" in price_df.columns:
            colors = [
                "green" if c >= o else "red"
                for c, o in zip(price_df["Close"], price_df.get("Open", price_df["Close"]))
            ]
            fig.add_trace(
                go.Bar(x=price_df.index, y=price_df["Volume"], name="Volume", marker_color=colors),
                row=2, col=1,
            )

        fig.update_layout(
            title=f"{ticker} — Price & Indicators",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=600,
        )
        return fig

    def prediction_comparison(
        self,
        price_df: pd.DataFrame,
        predictions: dict[str, PredictionResult],
        ticker: str,
    ) -> go.Figure:
        """Compare model predictions against actual price history."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=price_df.index[-60:],
                y=price_df["Close"].iloc[-60:],
                name="Actual Close",
                line={"color": "white", "width": 2},
            )
        )

        colors = ["blue", "orange", "green", "red", "purple"]
        for i, (model_name, pred) in enumerate(predictions.items()):
            last_date = price_df.index[-1]
            future_date = last_date + pd.Timedelta(days=pred.horizon_days)
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=[last_date, future_date],
                    y=[pred.current_price, pred.predicted_price],
                    name=f"{model_name} ({pred.predicted_return:+.1%})",
                    mode="lines+markers",
                    line={"color": color, "dash": "dash"},
                    marker={"size": 10},
                )
            )

        fig.update_layout(
            title=f"{ticker} — Prediction Comparison",
            template="plotly_dark",
            height=500,
            hovermode="x unified",
        )
        return fig

    def sentiment_timeline(
        self,
        sentiment_df: pd.DataFrame,
        ticker: str,
    ) -> go.Figure:
        """Sentiment compound score timeline with positive/negative bands."""
        if sentiment_df.empty or "compound" not in sentiment_df.columns:
            fig = go.Figure()
            fig.update_layout(title=f"{ticker} — No Sentiment Data")
            return fig

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])

        # Compound score
        fig.add_trace(
            go.Scatter(
                x=sentiment_df.index,
                y=sentiment_df["compound"],
                name="Compound",
                fill="tozeroy",
                line={"color": "cyan"},
            ),
            row=1, col=1,
        )

        # Post count
        if "post_count" in sentiment_df.columns:
            fig.add_trace(
                go.Bar(
                    x=sentiment_df.index,
                    y=sentiment_df["post_count"],
                    name="Post Count",
                    marker_color="rgba(100,200,255,0.5)",
                ),
                row=2, col=1,
            )

        fig.update_layout(
            title=f"{ticker} — Sentiment Timeline",
            template="plotly_dark",
            height=500,
        )
        return fig

    def backtest_equity_curve(
        self,
        equity_curve: pd.Series,
        ticker: str,
        strategy_name: str,
    ) -> go.Figure:
        """Equity curve chart with drawdown shading."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35])

        fig.add_trace(
            go.Scatter(
                x=equity_curve.index, y=equity_curve,
                name="Portfolio Value", line={"color": "lime"},
            ),
            row=1, col=1,
        )

        # Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        fig.add_trace(
            go.Scatter(
                x=drawdown.index, y=drawdown,
                name="Drawdown %", fill="tozeroy",
                line={"color": "red"},
            ),
            row=2, col=1,
        )

        fig.update_layout(
            title=f"{ticker} — {strategy_name} Equity Curve",
            template="plotly_dark",
            height=500,
        )
        return fig
