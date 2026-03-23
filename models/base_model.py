"""Base model abstraction: PredictionResult and BaseModel ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    """Immutable prediction output from any model."""

    ticker: str
    model_name: str
    horizon_days: int
    predicted_price: float
    predicted_return: float       # fractional (e.g. 0.05 = +5%)
    confidence: Optional[float]   # 0–1 prediction confidence
    direction_prob_up: Optional[float]  # probability of price increase
    current_price: float
    predicted_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base for all prediction models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model. Returns self."""
        ...

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        ticker: str,
        current_price: float,
        horizon_days: int,
    ) -> PredictionResult:
        """Generate a single PredictionResult for the most recent row."""
        ...

    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...
