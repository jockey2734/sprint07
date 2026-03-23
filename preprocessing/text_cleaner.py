"""Korean/English text cleaning utilities."""

from __future__ import annotations

import re
from typing import Sequence


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_SPECIAL_RE = re.compile(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?%$&]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str, language: str = "en") -> str:
    """Clean a single text string.

    Args:
        text: Raw text.
        language: "ko" for Korean, "en" for English.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)

    if language == "ko":
        # Keep Korean characters, alphanumeric, basic punctuation
        text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?%]", " ", text)
    else:
        text = _SPECIAL_RE.sub(" ", text)

    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_batch(texts: Sequence[str], language: str = "en") -> tuple[str, ...]:
    """Clean a sequence of texts, returning a new immutable tuple."""
    return tuple(clean_text(t, language) for t in texts)


def combine_title_body(title: str, body: str, language: str = "en") -> str:
    """Combine title and body with language-appropriate separator."""
    combined = f"{title}. {body}" if body else title
    return clean_text(combined, language)
