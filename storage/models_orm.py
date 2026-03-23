"""SQLAlchemy 2.0 ORM models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class RawPostORM(Base):
    __tablename__ = "raw_posts"
    __table_args__ = (UniqueConstraint("source", "post_id", name="uq_post"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)
    post_id = Column(String(200), nullable=False)
    ticker = Column(String(20), nullable=False)
    title = Column(Text, nullable=True)
    body = Column(Text, nullable=True)
    author = Column(String(200), nullable=True)
    upvotes = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=False)
    collected_at = Column(DateTime, default=datetime.utcnow)
    language = Column(String(10), nullable=True)


class SentimentResultORM(Base):
    __tablename__ = "sentiment_results"
    __table_args__ = (UniqueConstraint("ticker", "date", "source", name="uq_sentiment"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False)
    source = Column(String(50), nullable=False)
    positive = Column(Float, nullable=False)
    negative = Column(Float, nullable=False)
    neutral = Column(Float, nullable=False)
    compound = Column(Float, nullable=False)
    post_count = Column(Integer, default=0)
    weighted_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PriceDataORM(Base):
    __tablename__ = "price_data"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_price"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    adj_close = Column(Float, nullable=True)


class PredictionORM(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("ticker", "model_name", "horizon_days", "predicted_at", name="uq_pred"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    model_name = Column(String(100), nullable=False)
    horizon_days = Column(Integer, nullable=False)
    predicted_price = Column(Float, nullable=False)
    predicted_return = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    direction_prob_up = Column(Float, nullable=True)
    predicted_at = Column(DateTime, default=datetime.utcnow)
    current_price = Column(Float, nullable=True)


class BacktestResultORM(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    strategy = Column(String(100), nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_return = Column(Float, nullable=True)
    n_trades = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
