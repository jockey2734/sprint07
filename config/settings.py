"""Pydantic v2 AppConfig — single source of truth for all settings."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class NaverCrawlerConfig(BaseModel):
    enabled: bool = True
    max_pages: int = 5
    ttl_seconds: int = 3600


class RedditCrawlerConfig(BaseModel):
    enabled: bool = True
    subreddits: list[str] = Field(
        default=["stocks", "investing", "wallstreetbets", "StockMarket", "Korea_stocks"]
    )
    limit: int = 100
    ttl_seconds: int = 3600


class NewsCrawlerConfig(BaseModel):
    enabled: bool = True
    max_articles: int = 50
    ttl_seconds: int = 3600


class FearGreedConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 3600


class CrawlersConfig(BaseModel):
    naver: NaverCrawlerConfig = Field(default_factory=NaverCrawlerConfig)
    reddit: RedditCrawlerConfig = Field(default_factory=RedditCrawlerConfig)
    news: NewsCrawlerConfig = Field(default_factory=NewsCrawlerConfig)
    fear_greed: FearGreedConfig = Field(default_factory=FearGreedConfig)


class LinearModelConfig(BaseModel):
    short_term_days: list[int] = [5, 10, 20, 30]
    long_term_days: list[int] = [60, 90, 120, 180]
    alpha: float = 1.0


class PolynomialModelConfig(BaseModel):
    medium_term_days: list[int] = [14, 21, 42, 56]
    degree: int = 3
    alpha: float = 1.0


class LSTMModelConfig(BaseModel):
    sequence_length: int = 60
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    patience: int = 15


class XGBoostModelConfig(BaseModel):
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.01
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tree_method: str = "hist"
    device: str = "cuda"


class EnsembleWeights(BaseModel):
    lstm: float = 0.4
    xgboost: float = 0.3
    linear: float = 0.2
    polynomial: float = 0.1

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "EnsembleWeights":
        total = self.lstm + self.xgboost + self.linear + self.polynomial
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total:.4f}")
        return self


class EnsembleModelConfig(BaseModel):
    weights: EnsembleWeights = Field(default_factory=EnsembleWeights)


class ModelsConfig(BaseModel):
    linear: LinearModelConfig = Field(default_factory=LinearModelConfig)
    polynomial: PolynomialModelConfig = Field(default_factory=PolynomialModelConfig)
    lstm: LSTMModelConfig = Field(default_factory=LSTMModelConfig)
    xgboost: XGBoostModelConfig = Field(default_factory=XGBoostModelConfig)
    ensemble: EnsembleModelConfig = Field(default_factory=EnsembleModelConfig)


class BacktestingConfig(BaseModel):
    n_splits: int = 5
    train_ratio: float = 0.7
    threshold_buy: float = 0.02
    threshold_sell: float = -0.01
    initial_capital: float = 10000.0
    commission: float = 0.001


class TechnicalConfig(BaseModel):
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2
    sma_periods: list[int] = [5, 10, 20, 50, 200]
    ema_periods: list[int] = [12, 26]
    atr_period: int = 14


class SentimentFeatConfig(BaseModel):
    rolling_window: int = 7
    momentum_window: int = 3


class VolumeConfig(BaseModel):
    zscore_window: int = 20
    anomaly_threshold: float = 2.0


class FeaturesConfig(BaseModel):
    technical: TechnicalConfig = Field(default_factory=TechnicalConfig)
    sentiment: SentimentFeatConfig = Field(default_factory=SentimentFeatConfig)
    volume: VolumeConfig = Field(default_factory=VolumeConfig)


class StorageConfig(BaseModel):
    db_path: str = "output/stock_ai.db"
    cache_dir: str = "output/cache"
    reports_dir: str = "output/reports"
    default_ttl: int = 86400


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"


class TickerConfig(BaseModel):
    default: str = "AAPL"
    market: str = "US"  # US | KR | BOTH


class DatesConfig(BaseModel):
    start: str = "2020-01-01"
    end: Optional[str] = None


class AppConfig(BaseModel):
    """Root configuration model — all modules depend on this."""

    ticker: TickerConfig = Field(default_factory=TickerConfig)
    dates: DatesConfig = Field(default_factory=DatesConfig)
    crawlers: CrawlersConfig = Field(default_factory=CrawlersConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
