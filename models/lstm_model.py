"""PyTorch LSTM × 2 + MultiheadAttention model for time-series prediction."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from stock_ai.models.base_model import BaseModel, PredictionResult
from stock_ai.preprocessing.price_preprocessor import LSTMSequenceData


class _LSTMAttentionNet(nn.Module):
    """2-layer LSTM with MultiheadAttention."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=min(num_heads, hidden_size),
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm(attn_out + lstm_out)  # residual
        context = attn_out[:, -1, :]  # last time step
        out = self.fc(context).squeeze(-1)
        return out, attn_weights


class LSTMModel(BaseModel):
    """LSTM + Attention model. Requires pre-built LSTMSequenceData."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        device: str | None = None,
        horizon_days: int = 5,
    ) -> None:
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._patience = patience
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._horizon = horizon_days
        self._net: Optional[_LSTMAttentionNet] = None
        self._seq_data: Optional[LSTMSequenceData] = None
        self._fitted = False

    @property
    def model_name(self) -> str:
        return f"lstm_{self._horizon}d"

    def is_fitted(self) -> bool:
        return self._fitted

    def fit_sequences(self, seq_data: LSTMSequenceData) -> "LSTMModel":
        """Train from pre-built LSTMSequenceData."""
        self._seq_data = seq_data
        X_t = torch.tensor(seq_data.X, dtype=torch.float32)
        # Normalize y to returns to stabilize training
        y_raw = seq_data.y
        self._y_mean = float(np.mean(y_raw))
        self._y_std = float(np.std(y_raw)) or 1.0
        y_norm = (y_raw - self._y_mean) / self._y_std
        y_t = torch.tensor(y_norm, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        n_val = max(1, int(len(dataset) * 0.1))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_ds, batch_size=self._batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self._batch_size, shuffle=False)

        input_size = seq_data.X.shape[2]
        self._net = _LSTMAttentionNet(
            input_size=input_size,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(1, self._epochs + 1):
            self._net.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                pred, _ = self._net(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self._net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    pred, _ = self._net(xb)
                    val_loss += criterion(pred, yb).item()
            val_loss /= max(1, len(val_loader))
            # Skip NaN val_loss (numerical instability) — count as no improvement
            if not np.isfinite(val_loss):
                no_improve += 1
            else:
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1

            if epoch % 10 == 0:
                logger.debug(
                    f"LSTM epoch {epoch}/{self._epochs}: "
                    f"train={train_loss/len(train_loader):.4f} val={val_loss:.4f}"
                )

            if no_improve >= self._patience:
                logger.info(f"LSTM early stopping at epoch {epoch}")
                break

        self._fitted = True
        logger.info(f"LSTM trained: best_val_loss={best_val_loss:.6f}, device={self._device}")
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMModel":
        """Not directly used — call fit_sequences() instead.
        Provided for interface compatibility.
        """
        logger.warning("LSTMModel.fit() called without sequence data — no-op. Use fit_sequences().")
        return self

    def predict(
        self,
        X: pd.DataFrame,
        ticker: str,
        current_price: float,
        horizon_days: int | None = None,
    ) -> PredictionResult:
        raise NotImplementedError("Use predict_from_sequence() for LSTM predictions")

    def predict_from_sequence(
        self,
        last_sequence: np.ndarray,
        ticker: str,
        current_price: float,
        horizon_days: int | None = None,
    ) -> PredictionResult:
        """Generate prediction from last sequence window.

        Args:
            last_sequence: np.ndarray of shape (seq_len, n_features).
        """
        if not self.is_fitted() or self._net is None:
            raise RuntimeError("LSTMModel not fitted")

        horizon = horizon_days or self._horizon
        self._net.eval()
        with torch.no_grad():
            x = torch.tensor(last_sequence[np.newaxis], dtype=torch.float32).to(self._device)
            pred, attn = self._net(x)
            pred_norm = float(pred.cpu().item())

        # Denormalize: prediction was trained on normalized y
        y_mean = getattr(self, "_y_mean", current_price)
        y_std = getattr(self, "_y_std", current_price * 0.1)
        predicted_price_raw = pred_norm * y_std + y_mean

        if not np.isfinite(predicted_price_raw) or predicted_price_raw <= 0:
            # Fallback: use current price as prediction
            predicted_price_raw = current_price
        predicted_return = (predicted_price_raw - current_price) / max(current_price, 1e-9)
        predicted_price = predicted_price_raw

        return PredictionResult(
            ticker=ticker,
            model_name=self.model_name,
            horizon_days=horizon,
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence=0.7,  # placeholder; derive from ensemble variance
            direction_prob_up=1.0 if predicted_return > 0 else 0.0,
            current_price=current_price,
            metadata={"device": str(self._device)},
        )
