"""TTL file cache using pickle + SHA-256 keys."""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class FileCache:
    """Simple file-based cache with per-entry TTL support."""

    def __init__(self, cache_dir: str = "output/cache", default_ttl: int = 86400) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key_to_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._dir / f"{digest}.pkl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if missing / expired."""
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                entry: dict = pickle.load(f)
            if time.time() > entry["expires_at"]:
                path.unlink(missing_ok=True)
                logger.debug(f"Cache expired: {key[:40]}")
                return None
            logger.debug(f"Cache hit: {key[:40]}")
            return entry["value"]
        except Exception as exc:
            logger.warning(f"Cache read error ({key[:40]}): {exc}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value with TTL (seconds). Uses default_ttl if ttl is None."""
        path = self._key_to_path(key)
        entry = {
            "value": value,
            "expires_at": time.time() + (ttl if ttl is not None else self._default_ttl),
            "key_preview": key[:80],
        }
        try:
            with open(path, "wb") as f:
                pickle.dump(entry, f)
            logger.debug(f"Cache set: {key[:40]}")
        except Exception as exc:
            logger.warning(f"Cache write error ({key[:40]}): {exc}")

    def invalidate(self, key: str) -> None:
        """Remove a single cache entry."""
        path = self._key_to_path(key)
        path.unlink(missing_ok=True)

    def clear_all(self) -> int:
        """Delete all cache files. Returns count deleted."""
        count = 0
        for p in self._dir.glob("*.pkl"):
            p.unlink(missing_ok=True)
            count += 1
        logger.info(f"Cache cleared: {count} entries deleted")
        return count
