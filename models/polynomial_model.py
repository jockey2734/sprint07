"""Polynomial Ridge regression for mid-term (2-8 weeks) forecasting."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from stock_ai.models.base_model import BaseModel, PredictionResult


class PolynomialRegressionModel(BaseModel):
    """Polynomial (degree=3) Ridge regression for mid-term prediction."""

    def __init__(
        self, horizon_days: int = 21, degree: int = 3, alpha: float = 1.0
    ) -> None:
        self._horizon = horizon_days
        self._degree = degree
        self._alpha = alpha
        self._pipeline: Optional[Pipeline] = None

    @property
    def model_name(self) -> str:
        return f"polynomial_{self._horizon}d"

    def is_fitted(self) -> bool:
        return self._pipeline is not None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PolynomialRegressionModel":
        # Limit features to avoid combinatorial explosion
        top_cols = X.columns[:min(20, len(X.columns))]
        X_top = X[top_cols]

        self._pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=self._degree, include_bias=False)),
            ("ridge", Ridge(alpha=self._alpha)),
        ])
        self._pipeline.fit(X_top, y)
        self._top_cols = list(top_cols)
        logger.info(f"PolynomialModel {self.model_name}: fitted on {len(X)} samples (deg={self._degree})")
        return self

    def predict(
        self,
        X: pd.DataFrame,
        ticker: str,
        current_price: float,
        horizon_days: int | None = None,
    ) -> PredictionResult:
        if not self.is_fitted():
            raise RuntimeError(f"{self.model_name} has not been fitted")

        horizon = horizon_days or self._horizon
        top_cols = [c for c in self._top_cols if c in X.columns]
        last_X = X[top_cols].iloc[[-1]]
        predicted_return = float(self._pipeline.predict(last_X)[0])
        predicted_price = current_price * (1 + predicted_return)
        confidence = float(max(0.0, min(1.0, 1.0 / (1.0 + abs(predicted_return) * 8))))

        return PredictionResult(
            ticker=ticker,
            model_name=self.model_name,
            horizon_days=horizon,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence=confidence,
            direction_prob_up=1.0 if predicted_return > 0 else 0.0,
            current_price=current_price,
            metadata={"degree": self._degree, "alpha": self._alpha},
        )
