"""10-step pipeline orchestrator: end-to-end stock analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from stock_ai.config.settings import AppConfig
from stock_ai.crawlers.base import FearGreedData, RawPost
from stock_ai.crawlers.fear_greed import FearGreedCrawler
from stock_ai.crawlers.naver_finance import NaverFinanceCrawler
from stock_ai.crawlers.news_crawler import NewsCrawler
from stock_ai.crawlers.reddit import RedditCrawler
from stock_ai.features.feature_builder import FeatureBuilder, FeatureDataset
from stock_ai.features.fundamental_features import FundamentalFeatureBuilder
from stock_ai.features.sentiment_features import SentimentFeatureBuilder
from stock_ai.features.technical_indicators import TechnicalIndicators
from stock_ai.fundamentals.news_fetcher import FundamentalsNewsFetcher
from stock_ai.fundamentals.ratios import RatiosCalculator
from stock_ai.fundamentals.statement_fetcher import StatementFetcher
from stock_ai.market_data.price_fetcher import PriceFetcher
from stock_ai.market_data.volume_analyzer import VolumeAnalyzer
from stock_ai.models.base_model import PredictionResult
from stock_ai.models.ensemble_model import EnsembleModel, EnsembleWeights
from stock_ai.models.linear_model import LinearRegressionModel
from stock_ai.models.lstm_model import LSTMModel
from stock_ai.models.polynomial_model import PolynomialRegressionModel
from stock_ai.models.xgboost_model import XGBoostModel
from stock_ai.preprocessing.data_merger import DataMerger
from stock_ai.preprocessing.price_preprocessor import PricePreprocessor
from stock_ai.preprocessing.text_cleaner import combine_title_body
from stock_ai.sentiment.aggregator import SentimentAggregator
from stock_ai.sentiment.en_analyzer import EnglishSentimentAnalyzer
from stock_ai.sentiment.ko_analyzer import KoreanSentimentAnalyzer
from stock_ai.backtesting.engine import BacktestEngine, BacktestResult
from stock_ai.backtesting.strategy import ThresholdStrategy
from stock_ai.storage.cache import FileCache
from stock_ai.storage.database import create_db_engine, get_session
from stock_ai.storage.models_orm import PredictionORM, BacktestResultORM
from stock_ai.visualization.chart_builder import ChartBuilder
from stock_ai.visualization.report_generator import ReportGenerator


@dataclass(frozen=True)
class PipelineResult:
    """Immutable container for complete pipeline output."""

    ticker: str
    price_df: pd.DataFrame
    merged_df: pd.DataFrame
    predictions: dict[str, PredictionResult]
    backtest: Optional[BacktestResult]
    report_path: Optional[Path]
    warnings: tuple[str, ...]
    completed_at: datetime = field(default_factory=datetime.utcnow)


class StockPipeline:
    """10-step end-to-end stock analysis pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._cache = FileCache(
            cache_dir=config.storage.cache_dir,
            default_ttl=config.storage.default_ttl,
        )
        self._engine = create_db_engine(config.storage.db_path)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        ticker: str,
        skip_crawl: bool = False,
        skip_models: bool = False,
    ) -> PipelineResult:
        warnings: list[str] = []
        cfg = self._config

        logger.info(f"=== Pipeline START: {ticker} ===")

        # Step 1: CRAWL
        posts, fear_greed = self._step1_crawl(ticker, skip_crawl, warnings)

        # Step 2: FETCH price data
        price_df = self._step2_fetch_price(ticker, cfg)

        if price_df.empty:
            warnings.append(f"No price data for {ticker} — pipeline aborted")
            return PipelineResult(
                ticker=ticker,
                price_df=price_df,
                merged_df=pd.DataFrame(),
                predictions={},
                backtest=None,
                report_path=None,
                warnings=tuple(warnings),
            )

        # Step 2.5: FUNDAMENTALS
        fundamental_df = self._step25_fundamentals(ticker, warnings)

        # Step 3: PREPROCESS
        preprocessor = PricePreprocessor()
        price_clean = preprocessor.clean(price_df)

        # Clean post texts for sentiment
        ko_posts = [p for p in posts if p.language == "ko"]
        en_posts = [p for p in posts if p.language == "en"]

        # Step 4: SENTIMENT
        sentiment_df = self._step4_sentiment(ko_posts, en_posts, warnings)

        # Step 5: FEATURES
        merged_df = self._step5_features(
            price_clean, sentiment_df, fear_greed, fundamental_df, ticker, cfg
        )

        if skip_models:
            logger.info("Skipping model training (skip_models=True)")
            return PipelineResult(
                ticker=ticker,
                price_df=price_df,
                merged_df=merged_df,
                predictions={},
                backtest=None,
                report_path=None,
                warnings=tuple(warnings),
            )

        # Step 6: TRAIN models
        models_dict, lstm_seq = self._step6_train(merged_df, ticker, cfg, warnings)

        # Step 7: PREDICT
        current_price = float(price_df["Close"].iloc[-1])
        predictions = self._step7_predict(
            merged_df, models_dict, lstm_seq, ticker, current_price, cfg
        )

        # Step 8: BACKTEST
        backtest = self._step8_backtest(merged_df, predictions, ticker, cfg)

        # Step 9: VISUALIZE + REPORT
        report_path = self._step9_visualize(
            ticker, price_df, merged_df, sentiment_df, predictions, backtest, warnings, cfg
        )

        # Step 10: STORE
        self._step10_store(ticker, predictions, backtest)

        logger.info(f"=== Pipeline DONE: {ticker} — {len(predictions)} predictions ===")
        return PipelineResult(
            ticker=ticker,
            price_df=price_df,
            merged_df=merged_df,
            predictions=predictions,
            backtest=backtest,
            report_path=report_path,
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _step1_crawl(
        self, ticker: str, skip: bool, warnings: list[str]
    ) -> tuple[tuple[RawPost, ...], Optional[FearGreedData]]:
        if skip:
            return (), None

        logger.info("Step 1: Crawling community posts...")
        all_posts: list[RawPost] = []
        cfg = self._config

        # Naver Finance (Korean tickers only)
        if cfg.crawlers.naver.enabled and (ticker.endswith(".KS") or ticker.endswith(".KQ")):
            crawler = NaverFinanceCrawler(self._cache, cfg.crawlers.naver.max_pages)
            all_posts.extend(crawler.crawl(ticker))

        # Reddit (optional)
        if cfg.crawlers.reddit.enabled:
            reddit = RedditCrawler(
                subreddits=cfg.crawlers.reddit.subreddits,
                limit=cfg.crawlers.reddit.limit,
                cache=self._cache,
            )
            if reddit.is_available():
                all_posts.extend(reddit.crawl(ticker))
            else:
                warnings.append("Reddit crawler unavailable (no credentials)")

        # News
        if cfg.crawlers.news.enabled:
            news = NewsCrawler(self._cache, cfg.crawlers.news.max_articles)
            all_posts.extend(news.crawl(ticker))

        # Fear & Greed
        fear_greed = None
        if cfg.crawlers.fear_greed.enabled:
            fg_crawler = FearGreedCrawler(self._cache)
            fear_greed = fg_crawler.fetch()

        logger.info(f"Step 1 complete: {len(all_posts)} posts crawled")
        return tuple(all_posts), fear_greed

    def _step2_fetch_price(self, ticker: str, cfg: AppConfig) -> pd.DataFrame:
        logger.info("Step 2: Fetching price data...")
        fetcher = PriceFetcher(self._cache)
        df = fetcher.fetch(ticker, start=cfg.dates.start, end=cfg.dates.end)
        logger.info(f"Step 2 complete: {len(df)} price rows")
        return df

    def _step25_fundamentals(
        self, ticker: str, warnings: list[str]
    ) -> Optional[pd.DataFrame]:
        logger.info("Step 2.5: Fetching fundamental data...")
        try:
            stmt_fetcher = StatementFetcher(self._cache)
            statements = stmt_fetcher.fetch(ticker)
            ratios = RatiosCalculator(self._cache)
            fund_df = ratios.compute_time_series(statements)
            logger.info(f"Step 2.5 complete: {fund_df.shape} fundamentals")
            return fund_df if not fund_df.empty else None
        except Exception as exc:
            warnings.append(f"Fundamentals fetch failed: {exc}")
            return None

    def _step4_sentiment(
        self,
        ko_posts: list[RawPost],
        en_posts: list[RawPost],
        warnings: list[str],
    ) -> pd.DataFrame:
        logger.info("Step 4: Running sentiment analysis...")
        aggregator = SentimentAggregator()
        all_posts: list = []
        all_sentiments: list = []

        if ko_posts:
            ko_analyzer = KoreanSentimentAnalyzer()
            ko_texts = [combine_title_body(p.title, p.body, "ko") for p in ko_posts]
            ko_sentiments = list(ko_analyzer.analyze_batch(ko_texts))
            all_posts.extend(ko_posts)
            all_sentiments.extend(ko_sentiments)

        if en_posts:
            en_analyzer = EnglishSentimentAnalyzer()
            en_texts = [combine_title_body(p.title, p.body, "en") for p in en_posts]
            en_sentiments = list(en_analyzer.analyze_batch(en_texts))
            all_posts.extend(en_posts)
            all_sentiments.extend(en_sentiments)

        if not all_posts:
            logger.warning("Step 4: No posts to analyze sentiment")
            return pd.DataFrame()

        daily_df = aggregator.aggregate(all_posts, all_sentiments)
        logger.info(f"Step 4 complete: {len(daily_df)} daily sentiment rows")
        return daily_df

    def _step5_features(
        self,
        price_clean: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        fear_greed: Optional[FearGreedData],
        fundamental_df: Optional[pd.DataFrame],
        ticker: str,
        cfg: AppConfig,
    ) -> pd.DataFrame:
        logger.info("Step 5: Building features...")
        tc = cfg.features.technical

        # Technical indicators
        tech = TechnicalIndicators(
            rsi_period=tc.rsi_period,
            macd_fast=tc.macd_fast,
            macd_slow=tc.macd_slow,
            macd_signal=tc.macd_signal,
            bb_period=tc.bb_period,
            bb_std=tc.bb_std,
            sma_periods=tc.sma_periods,
            ema_periods=tc.ema_periods,
            atr_period=tc.atr_period,
        )
        price_with_tech = tech.compute_all(price_clean)

        # Volume anomaly
        vol_analyzer = VolumeAnalyzer(
            window=cfg.features.volume.zscore_window,
            threshold=cfg.features.volume.anomaly_threshold,
        )
        price_with_tech = vol_analyzer.compute_zscore(price_with_tech)

        # Merge all data
        merger = DataMerger()
        merged = merger.merge(price_with_tech, sentiment_df, fear_greed, fundamental_df)

        # Sentiment features
        sent_builder = SentimentFeatureBuilder(
            rolling_window=cfg.features.sentiment.rolling_window,
            momentum_window=cfg.features.sentiment.momentum_window,
        )
        merged = sent_builder.build(merged)

        logger.info(f"Step 5 complete: merged shape {merged.shape}")
        return merged

    def _step6_train(
        self,
        merged_df: pd.DataFrame,
        ticker: str,
        cfg: AppConfig,
        warnings: list[str],
    ) -> tuple[dict, Optional[object]]:
        logger.info("Step 6: Training models...")
        models: dict = {}

        # Build feature dataset
        fb = FeatureBuilder(horizon_days=5, scale=True)
        try:
            dataset = fb.build(merged_df, ticker)
        except Exception as exc:
            warnings.append(f"Feature building failed: {exc}")
            return {}, None

        X, y = dataset.X, dataset.y
        if len(X) < 30:
            warnings.append(f"Insufficient training data: {len(X)} samples")
            return {}, None

        # Linear (short-term)
        try:
            lin = LinearRegressionModel(horizon_days=5, alpha=cfg.models.linear.alpha)
            lin.fit(X, y)
            models["linear"] = lin
        except Exception as exc:
            warnings.append(f"Linear model training failed: {exc}")

        # Polynomial (mid-term)
        try:
            poly = PolynomialRegressionModel(
                horizon_days=21,
                degree=cfg.models.polynomial.degree,
                alpha=cfg.models.polynomial.alpha,
            )
            poly.fit(X, y)
            models["polynomial"] = poly
        except Exception as exc:
            warnings.append(f"Polynomial model training failed: {exc}")

        # XGBoost
        try:
            xgb = XGBoostModel(
                horizon_days=5,
                n_estimators=cfg.models.xgboost.n_estimators,
                max_depth=cfg.models.xgboost.max_depth,
                learning_rate=cfg.models.xgboost.learning_rate,
            )
            xgb.fit(X, y)
            models["xgboost"] = xgb
        except Exception as exc:
            warnings.append(f"XGBoost training failed: {exc}")

        # LSTM (requires sequence data)
        lstm_seq = None
        try:
            preprocessor = PricePreprocessor()
            feature_cols = list(dataset.feature_names)
            seq_data = preprocessor.create_lstm_sequences(
                merged_df,
                feature_cols=feature_cols,
                seq_length=cfg.models.lstm.sequence_length,
            )
            lstm = LSTMModel(
                hidden_size=cfg.models.lstm.hidden_size,
                num_layers=cfg.models.lstm.num_layers,
                dropout=cfg.models.lstm.dropout,
                learning_rate=cfg.models.lstm.learning_rate,
                epochs=cfg.models.lstm.epochs,
                batch_size=cfg.models.lstm.batch_size,
                patience=cfg.models.lstm.patience,
            )
            lstm.fit_sequences(seq_data)
            models["lstm_model"] = lstm
            lstm_seq = (lstm, seq_data)
        except Exception as exc:
            warnings.append(f"LSTM training failed: {exc}")

        logger.info(f"Step 6 complete: {list(models.keys())} trained")
        return models, lstm_seq

    def _step7_predict(
        self,
        merged_df: pd.DataFrame,
        models: dict,
        lstm_seq: Optional[tuple],
        ticker: str,
        current_price: float,
        cfg: AppConfig,
    ) -> dict[str, PredictionResult]:
        logger.info("Step 7: Generating predictions...")
        fb = FeatureBuilder(horizon_days=5, scale=True)
        predictions: dict[str, PredictionResult] = {}

        try:
            dataset = fb.build(merged_df, ticker)
        except Exception:
            return predictions

        X = dataset.X

        # Individual model predictions
        for name, model in models.items():
            if name == "lstm_model":
                continue
            try:
                pred = model.predict(X, ticker, current_price)
                predictions[name] = pred
            except Exception as exc:
                logger.warning(f"Prediction failed for {name}: {exc}")

        # LSTM prediction
        lstm_result = None
        if lstm_seq is not None:
            try:
                lstm_model, seq_data = lstm_seq
                last_seq = seq_data.X[-1]  # last sequence window
                lstm_result = lstm_model.predict_from_sequence(
                    last_seq, ticker, current_price, horizon_days=5
                )
                predictions["lstm"] = lstm_result
            except Exception as exc:
                logger.warning(f"LSTM prediction failed: {exc}")

        # Ensemble
        try:
            ensemble_models = {k: v for k, v in models.items() if k != "lstm_model"}
            ens = EnsembleModel(
                weights=EnsembleWeights(
                    lstm=cfg.models.ensemble.weights.lstm,
                    xgboost=cfg.models.ensemble.weights.xgboost,
                    linear=cfg.models.ensemble.weights.linear,
                    polynomial=cfg.models.ensemble.weights.polynomial,
                ),
                horizon_days=5,
            )
            ind_preds = ens.predict_all(X, ticker, current_price, ensemble_models, lstm_result)
            if ind_preds:
                ens_pred = ens.predict_ensemble(ind_preds, ticker, current_price)
                predictions["ensemble"] = ens_pred
        except Exception as exc:
            logger.warning(f"Ensemble prediction failed: {exc}")

        logger.info(f"Step 7 complete: {list(predictions.keys())}")
        return predictions

    def _step8_backtest(
        self,
        merged_df: pd.DataFrame,
        predictions: dict[str, PredictionResult],
        ticker: str,
        cfg: AppConfig,
    ) -> Optional[BacktestResult]:
        logger.info("Step 8: Backtesting...")
        try:
            bt = cfg.backtesting
            engine = BacktestEngine(
                n_splits=bt.n_splits,
                train_ratio=bt.train_ratio,
                initial_capital=bt.initial_capital,
                commission=bt.commission,
            )
            strategy = ThresholdStrategy(
                threshold_buy=bt.threshold_buy,
                threshold_sell=bt.threshold_sell,
            )
            result = engine.run_walk_forward(merged_df, strategy, ticker=ticker)
            logger.info(f"Step 8 complete: Sharpe={result.avg_sharpe:.2f}")
            return result
        except Exception as exc:
            logger.warning(f"Backtest failed: {exc}")
            return None

    def _step9_visualize(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        predictions: dict[str, PredictionResult],
        backtest: Optional[BacktestResult],
        warnings: list[str],
        cfg: AppConfig,
    ) -> Optional[Path]:
        logger.info("Step 9: Generating visualizations...")
        try:
            builder = ChartBuilder()
            charts: dict = {}

            if not price_df.empty:
                charts["price"] = builder.candlestick_with_indicators(
                    merged_df if not merged_df.empty else price_df, ticker
                )

            if predictions:
                charts["predictions"] = builder.prediction_comparison(price_df, predictions, ticker)

            if not sentiment_df.empty:
                charts["sentiment"] = builder.sentiment_timeline(sentiment_df, ticker)

            if backtest and backtest.folds:
                fold_equity = backtest.folds[-1].equity_curve
                charts["backtest"] = builder.backtest_equity_curve(
                    fold_equity, ticker, backtest.strategy_name
                )

            reporter = ReportGenerator(cfg.storage.reports_dir)
            path = reporter.generate(
                ticker=ticker,
                charts=charts,
                predictions=predictions,
                backtest=backtest,
                warnings=warnings,
            )
            logger.info(f"Step 9 complete: {path}")
            return path
        except Exception as exc:
            logger.error(f"Report generation failed: {exc}")
            return None

    def _step10_store(
        self,
        ticker: str,
        predictions: dict[str, PredictionResult],
        backtest: Optional[BacktestResult],
    ) -> None:
        logger.info("Step 10: Storing results...")
        try:
            import math
            with get_session(self._engine) as session:
                # Store predictions (skip any with NaN values)
                for pred in predictions.values():
                    if not math.isfinite(pred.predicted_price) or not math.isfinite(pred.predicted_return):
                        logger.warning(f"Skipping storage of {pred.model_name}: NaN prediction")
                        continue
                    orm = PredictionORM(
                        ticker=pred.ticker,
                        model_name=pred.model_name,
                        horizon_days=pred.horizon_days,
                        predicted_price=pred.predicted_price,
                        predicted_return=pred.predicted_return,
                        confidence=pred.confidence,
                        direction_prob_up=pred.direction_prob_up,
                        current_price=pred.current_price,
                    )
                    session.merge(orm)

                # Store backtest summary
                if backtest and backtest.folds:
                    bt_orm = BacktestResultORM(
                        ticker=ticker,
                        strategy=backtest.strategy_name,
                        sharpe_ratio=backtest.avg_sharpe,
                        max_drawdown=backtest.avg_max_drawdown,
                        win_rate=backtest.avg_win_rate,
                        total_return=backtest.avg_total_return,
                        n_trades=backtest.total_trades,
                    )
                    session.add(bt_orm)

            logger.info("Step 10 complete: results stored in SQLite")
        except Exception as exc:
            logger.error(f"Storage failed: {exc}")
