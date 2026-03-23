"""Z-score based volume anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class VolumeAnomaly:
    """Detected volume anomaly event."""

    date: pd.Timestamp
    volume: float
    z_score: float
    direction: str  # "surge" | "plunge"


class VolumeAnalyzer:
    """Detects unusual trading volume using rolling Z-score."""

    def __init__(self, window: int = 20, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def compute_zscore(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Add volume Z-score column to price DataFrame.

        Returns new DataFrame with added columns:
          - volume_zscore
          - volume_surge (bool)
          - volume_plunge (bool)
        """
        if "Volume" not in price_df.columns:
            logger.warning("VolumeAnalyzer: 'Volume' column not found")
            return price_df

        df = price_df.copy()
        roll_mean = df["Volume"].rolling(self._window).mean()
        roll_std = df["Volume"].rolling(self._window).std()
        df["volume_zscore"] = (df["Volume"] - roll_mean) / (roll_std + 1e-9)
        df["volume_surge"] = df["volume_zscore"] > self._threshold
        df["volume_plunge"] = df["volume_zscore"] < -self._threshold
        return df

    def detect_anomalies(self, price_df: pd.DataFrame) -> tuple[VolumeAnomaly, ...]:
        """Return tuple of VolumeAnomaly for dates exceeding threshold."""
        df = self.compute_zscore(price_df)
        if "volume_zscore" not in df.columns:
            return ()

        anomalies: list[VolumeAnomaly] = []
        mask = df["volume_zscore"].abs() > self._threshold
        for idx, row in df[mask].iterrows():
            z = float(row["volume_zscore"])
            anomalies.append(
                VolumeAnomaly(
                    date=idx,
                    volume=float(row["Volume"]),
                    z_score=z,
                    direction="surge" if z > 0 else "plunge",
                )
            )
        logger.info(f"VolumeAnalyzer: {len(anomalies)} anomalies detected")
        return tuple(anomalies)
