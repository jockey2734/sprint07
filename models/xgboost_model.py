"""XGBoost model: price return regression + direction classification."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from stock_ai.models.base_model import BaseModel, PredictionResult

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed")


class XGBoostModel(BaseModel):
    """XGBoost for return regression with optional GPU acceleration."""

    def __init__(
        self,
        horizon_days: int = 5,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.01,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        tree_method: str = "hist",
        device: str = "cuda",
    ) -> None:
        self._horizon = horizon_days
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": tree_method,
            "device": device,
            "random_state": 42,
            "verbosity": 0,
        }
        self._regressor: Optional[object] = None
        self._classifier: Optional[object] = None
        self._feature_names: Optional[list[str]] = None

    @property
    def model_name(self) -> str:
        return f"xgboost_{self._horizon}d"

    def is_fitted(self) -> bool:
        return self._regressor is not None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost not installed")

        self._feature_names = list(X.columns)

        # Regression (return prediction)
        params_reg = {**self._params}
        try:
            self._regressor = xgb.XGBRegressor(**params_reg)
            self._regressor.fit(X, y)
        except Exception:
            # GPU may not be available — fallback to CPU
            params_cpu = {**self._params, "device": "cpu"}
            self._regressor = xgb.XGBRegressor(**params_cpu)
            self._regressor.fit(X, y)
            logger.warning("XGBoost: GPU not available, using CPU")

        # Classification (direction: 1=up, 0=down)
        y_dir = (y > 0).astype(int)
        params_cls = {**self._params}
        params_cls["objective"] = "binary:logistic"
        params_cls["eval_metric"] = "logloss"
        try:
            self._classifier = xgb.XGBClassifier(**params_cls)
            self._classifier.fit(X, y_dir)
        except Exception:
            params_cpu = {**params_cls, "device": "cpu"}
            self._classifier = xgb.XGBClassifier(**params_cpu)
            self._classifier.fit(X, y_dir)

        logger.info(f"XGBoost {self.model_name}: fitted on {len(X)} samples")
        return self

    def predict(
        self,
        X: pd.DataFrame,
        ticker: str,
        current_price: float,
        horizon_days: int | None = None,
    ) -> PredictionResult:
        if not self.is_fitted():
            raise RuntimeError(f"{self.model_name} not fitted")

        horizon = horizon_days or self._horizon
        last_X = X.iloc[[-1]]

        predicted_return = float(self._regressor.predict(last_X)[0])
        predicted_price = current_price * (1 + predicted_return)

        # Direction probability from classifier
        direction_prob_up = None
        if self._classifier is not None:
            try:
                proba = self._classifier.predict_proba(last_X)[0]
                direction_prob_up = float(proba[1])  # probability of class=1 (up)
            except Exception:
                pass

        # Feature importance (top feature as metadata)
        top_feature = None
        if self._feature_names and hasattr(self._regressor, "feature_importances_"):
            imp = self._regressor.feature_importances_
            top_idx = int(np.argmax(imp))
            top_feature = self._feature_names[top_idx]

        return PredictionResult(
            ticker=ticker,
            model_name=self.model_name,
            horizon_days=horizon,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence=direction_prob_up,
            direction_prob_up=direction_prob_up,
            current_price=current_price,
            metadata={"top_feature": top_feature},
        )

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances as Series."""
        if not self.is_fitted() or self._feature_names is None:
            return pd.Series(dtype=float)
        imp = getattr(self._regressor, "feature_importances_", None)
        if imp is None:
            return pd.Series(dtype=float)
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)
