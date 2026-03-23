"""Weighted ensemble combining LSTM, XGBoost, Linear, and Polynomial models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from stock_ai.models.base_model import BaseModel, PredictionResult
from stock_ai.models.linear_model import LinearRegressionModel
from stock_ai.models.polynomial_model import PolynomialRegressionModel
from stock_ai.models.xgboost_model import XGBoostModel


@dataclass(frozen=True)
class EnsembleWeights:
    lstm: float = 0.4
    xgboost: float = 0.3
    linear: float = 0.2
    polynomial: float = 0.1


class EnsembleModel:
    """Weighted ensemble of all prediction models."""

    def __init__(
        self,
        weights: Optional[EnsembleWeights] = None,
        horizon_days: int = 5,
    ) -> None:
        self._weights = weights or EnsembleWeights()
        self._horizon = horizon_days

    def predict_all(
        self,
        X: pd.DataFrame,
        ticker: str,
        current_price: float,
        models: dict,  # {"linear": LinearModel, "polynomial": PolyModel, "xgboost": XGBModel}
        lstm_result: Optional[PredictionResult] = None,
    ) -> dict[str, PredictionResult]:
        """Generate individual predictions from all available models."""
        results: dict[str, PredictionResult] = {}

        if lstm_result is not None:
            results["lstm"] = lstm_result

        for name, model in models.items():
            if model is not None and hasattr(model, "is_fitted") and model.is_fitted():
                try:
                    results[name] = model.predict(X, ticker, current_price)
                except Exception as exc:
                    logger.warning(f"Ensemble: {name} prediction failed: {exc}")

        return results

    def predict_ensemble(
        self,
        individual_predictions: dict[str, PredictionResult],
        ticker: str,
        current_price: float,
        horizon_days: int | None = None,
    ) -> PredictionResult:
        """Compute weighted average of individual model predictions."""
        horizon = horizon_days or self._horizon

        weight_map = {
            "lstm": self._weights.lstm,
            "xgboost": self._weights.xgboost,
            "linear": self._weights.linear,
            "polynomial": self._weights.polynomial,
        }

        import math
        # Filter to available predictions with finite values, normalize weights
        available = {
            k: v for k, v in individual_predictions.items()
            if k in weight_map and math.isfinite(v.predicted_return)
        }
        if not available:
            raise ValueError("No model predictions available for ensemble")

        total_weight = sum(weight_map[k] for k in available)
        if total_weight == 0:
            raise ValueError("Zero total weight in ensemble")

        weighted_return = sum(
            weight_map[k] * v.predicted_return / total_weight
            for k, v in available.items()
        )
        weighted_price = current_price * (1 + weighted_return)

        # Average direction probability
        dir_probs = [
            v.direction_prob_up for v in available.values()
            if v.direction_prob_up is not None
        ]
        direction_prob = float(np.mean(dir_probs)) if dir_probs else None

        # Confidence: std of predictions (lower std = higher confidence)
        returns = [v.predicted_return for v in available.values()]
        std_returns = float(np.std(returns)) if len(returns) > 1 else 0.0
        confidence = float(max(0.0, min(1.0, 1.0 - std_returns * 10)))

        logger.info(
            f"Ensemble prediction for {ticker}: return={weighted_return:.3f}, "
            f"price={weighted_price:.2f}, confidence={confidence:.2f}"
        )

        return PredictionResult(
            ticker=ticker,
            model_name="ensemble",
            horizon_days=horizon,
            predicted_price=weighted_price,
            predicted_return=weighted_return,
            confidence=confidence,
            direction_prob_up=direction_prob,
            current_price=current_price,
            metadata={
                "components": list(available.keys()),
                "weights_used": {k: weight_map[k] for k in available},
                "std_returns": std_returns,
            },
        )
