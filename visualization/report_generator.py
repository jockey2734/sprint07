"""HTML report generator: all charts + backtest summary in a single file."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from stock_ai.backtesting.engine import BacktestResult
from stock_ai.models.base_model import PredictionResult


class ReportGenerator:
    """Generates a self-contained HTML dashboard with all analysis."""

    def __init__(self, reports_dir: str = "output/reports") -> None:
        self._dir = Path(reports_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        ticker: str,
        charts: dict[str, go.Figure],
        predictions: dict[str, PredictionResult],
        backtest: Optional[BacktestResult] = None,
        warnings: list[str] | None = None,
    ) -> Path:
        """Generate and save HTML report.

        Returns:
            Path to the generated HTML file.
        """
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker.replace('.', '_')}_{date_str}.html"
        output_path = self._dir / filename

        chart_html = self._render_charts(charts)
        pred_html = self._render_predictions(predictions)
        backtest_html = self._render_backtest(backtest)
        warnings_html = self._render_warnings(warnings or [])

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock AI Report — {ticker}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #00d4ff; text-align: center; }}
        h2 {{ color: #7fb3d3; border-bottom: 1px solid #333; padding-bottom: 8px; }}
        .section {{ margin: 20px 0; padding: 20px; background: #16213e; border-radius: 12px; }}
        .pred-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 15px; }}
        .pred-card {{ background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }}
        .pred-card .return {{ font-size: 1.5em; font-weight: bold; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4757; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
        .metric {{ background: #0f3460; padding: 12px; border-radius: 8px; text-align: center; }}
        .metric .value {{ font-size: 1.3em; font-weight: bold; color: #00d4ff; }}
        .warning {{ background: #3d1a00; border-left: 4px solid #ff6b35; padding: 10px; margin: 8px 0; border-radius: 4px; }}
        .timestamp {{ color: #666; text-align: center; font-size: 0.85em; margin-top: 20px; }}
    </style>
</head>
<body>
    <h1>Stock Anticipation AI — {ticker}</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    {warnings_html}

    <div class="section">
        <h2>Model Predictions</h2>
        {pred_html}
    </div>

    {backtest_html}

    <div class="section">
        <h2>Charts</h2>
        {chart_html}
    </div>

    <p class="timestamp">⚠️ This report is for educational purposes only. Not financial advice.</p>
</body>
</html>"""

        output_path.write_text(html, encoding="utf-8")
        logger.info(f"Report saved: {output_path}")
        return output_path

    def _render_charts(self, charts: dict[str, go.Figure]) -> str:
        parts: list[str] = []
        for name, fig in charts.items():
            div_id = f"chart_{name}"
            chart_json = fig.to_json()
            parts.append(
                f'<div id="{div_id}"></div>'
                f'<script>Plotly.newPlot("{div_id}", {chart_json});</script>'
            )
        return "\n".join(parts)

    def _render_predictions(self, predictions: dict[str, PredictionResult]) -> str:
        if not predictions:
            return "<p>No predictions available.</p>"
        cards: list[str] = []
        for model_name, pred in predictions.items():
            ret_pct = pred.predicted_return * 100
            css_class = "positive" if ret_pct >= 0 else "negative"
            sign = "+" if ret_pct >= 0 else ""
            conf_str = f"{pred.confidence * 100:.0f}%" if pred.confidence else "N/A"
            cards.append(f"""
                <div class="pred-card">
                    <div><strong>{model_name}</strong></div>
                    <div>Horizon: {pred.horizon_days}d</div>
                    <div class="return {css_class}">{sign}{ret_pct:.2f}%</div>
                    <div>Target: ${pred.predicted_price:.2f}</div>
                    <div>Confidence: {conf_str}</div>
                </div>""")
        return f'<div class="pred-grid">{"".join(cards)}</div>'

    def _render_backtest(self, backtest: Optional[BacktestResult]) -> str:
        if backtest is None or not backtest.folds:
            return ""
        mdd_pct = backtest.avg_max_drawdown * 100
        ret_pct = backtest.avg_total_return * 100
        return f"""
        <div class="section">
            <h2>Backtest Results — {backtest.strategy_name}</h2>
            <div class="metric-grid">
                <div class="metric"><div>Sharpe Ratio</div><div class="value">{backtest.avg_sharpe:.2f}</div></div>
                <div class="metric"><div>Max Drawdown</div><div class="value">{mdd_pct:.1f}%</div></div>
                <div class="metric"><div>Win Rate</div><div class="value">{backtest.avg_win_rate:.1%}</div></div>
                <div class="metric"><div>Total Return</div><div class="value">{ret_pct:+.1f}%</div></div>
                <div class="metric"><div>Total Trades</div><div class="value">{backtest.total_trades}</div></div>
                <div class="metric"><div>Folds</div><div class="value">{len(backtest.folds)}</div></div>
            </div>
        </div>"""

    def _render_warnings(self, warnings: list[str]) -> str:
        if not warnings:
            return ""
        items = "".join(f'<div class="warning">⚠️ {w}</div>' for w in warnings)
        return f'<div class="section"><h2>Warnings</h2>{items}</div>'
