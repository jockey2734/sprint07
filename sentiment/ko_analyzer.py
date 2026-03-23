"""Korean financial sentiment analyzer using snunlp/KR-FinBert-SC."""

from __future__ import annotations

from typing import Sequence

import torch
from loguru import logger

from stock_ai.sentiment.base_analyzer import BaseSentimentAnalyzer, SentimentResult

_MODEL_NAME = "snunlp/KR-FinBert-SC"
_LABEL_MAP = {"positive": "positive", "negative": "negative", "neutral": "neutral"}


class KoreanSentimentAnalyzer(BaseSentimentAnalyzer):
    """Financial sentiment for Korean texts using KR-FinBert-SC.

    First run: ~400 MB model download from HuggingFace.
    """

    def __init__(self, batch_size: int = 32, device: str | None = None) -> None:
        self._batch_size = batch_size
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "text-classification",
                model=_MODEL_NAME,
                device=0 if self._device == "cuda" else -1,
                truncation=True,
                max_length=512,
                top_k=None,  # return all labels
            )
            logger.info(f"KR-FinBert loaded on {self._device}")
        except Exception as exc:
            logger.error(f"KR-FinBert load failed: {exc}")
            self._pipeline = None

    def is_available(self) -> bool:
        self._load()
        return self._pipeline is not None

    def analyze(self, text: str) -> SentimentResult:
        results = self.analyze_batch([text])
        return results[0]

    def analyze_batch(self, texts: Sequence[str]) -> tuple[SentimentResult, ...]:
        self._load()
        if self._pipeline is None:
            return tuple(self._fallback(t) for t in texts)

        outputs: list[SentimentResult] = []
        text_list = list(texts)

        for i in range(0, len(text_list), self._batch_size):
            batch = text_list[i : i + self._batch_size]
            try:
                raw = self._pipeline(batch)
                for text, item_scores in zip(batch, raw):
                    scores = {s["label"].lower(): s["score"] for s in item_scores}
                    pos = scores.get("positive", 0.0)
                    neg = scores.get("negative", 0.0)
                    neu = scores.get("neutral", 0.0)
                    compound = pos - neg
                    label = max(scores, key=scores.get)
                    outputs.append(
                        SentimentResult(
                            text=text[:100],
                            positive=pos,
                            negative=neg,
                            neutral=neu,
                            compound=compound,
                            label=label,
                            language="ko",
                            model=_MODEL_NAME,
                        )
                    )
            except Exception as exc:
                logger.warning(f"KR-FinBert batch error: {exc}")
                outputs.extend(self._fallback(t) for t in batch)

        return tuple(outputs)

    def _fallback(self, text: str) -> SentimentResult:
        return SentimentResult(
            text=text[:100],
            positive=0.33,
            negative=0.33,
            neutral=0.34,
            compound=0.0,
            label="neutral",
            language="ko",
            model="fallback",
        )
