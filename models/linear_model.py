"""Ridge Regression: short-term (5-30 days) and long-term (1-6 months) forecasting."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from stock_ai.models.base_model import BaseModel, PredictionResult


class LinearRegressionModel(BaseModel):
    """Ridge regression model for single-horizon prediction."""

    def __init__(self, horizon_days: int = 5, alpha: float = 1.0) -> None:
        self._horizon = horizon_days
        self._alpha = alpha
        self._model: Optional[Ridge] = None
        self._cv_score: Optional[float] = None

    @property
    def model_name(self) -> str:
        return f"linear_{self._horizon}d"

    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionModel":
        """Fit Ridge regression with TimeSeriesCV."""
        self._model = Ridge(alpha=self._alpha)
        self._model.fit(X, y)

        # Cross-validation score
        if len(X) >= 10:
            tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 5))
            scores = cross_val_score(
                Ridge(alpha=self._alpha), X, y, cv=tscv, scoring="neg_mean_squared_error"
            )
            self._cv_score = float(np.sqrt(-scores.mean()))
            logger.info(f"LinearModel {self.model_name}: fitted, CV RMSE={self._cv_score:.4f}")
        else:
            logger.info(f"LinearModel {self.model_name}: fitted ({len(X)} samples)")
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
        last_X = X.iloc[[-1]]
        predicted_return = float(self._model.predict(last_X)[0])
        predicted_price = current_price * (1 + predicted_return)

        # Confidence: inverse of training variance
        confidence = float(max(0.0, min(1.0, 1.0 / (1.0 + abs(predicted_return) * 10))))

        return PredictionResult(
            ticker=ticker,
            model_name=self.model_name,
            horizon_days=horizon,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence=confidence,
            direction_prob_up=1.0 if predicted_return > 0 else 0.0,
            current_price=current_price,
            metadata={"cv_rmse": self._cv_score, "alpha": self._alpha},
        )
