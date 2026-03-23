"""Shared sentiment types and base analyzer ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


@dataclass(frozen=True)
class SentimentResult:
    """Immutable result for a single text sentiment analysis."""

    text: str
    positive: float   # 0.0–1.0
    negative: float   # 0.0–1.0
    neutral: float    # 0.0–1.0
    compound: float   # -1.0 to +1.0 (positive - negative)
    label: str        # "positive" | "negative" | "neutral"
    language: str     # "ko" | "en"
    model: str        # model identifier


class BaseSentimentAnalyzer(ABC):
    """Abstract base for KR and EN sentiment analyzers."""

    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Analyze a single text string."""
        ...

    @abstractmethod
    def analyze_batch(self, texts: Sequence[str]) -> tuple[SentimentResult, ...]:
        """Batch analyze texts (GPU-optimized implementations override this)."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the model is loaded and ready."""
        ...
