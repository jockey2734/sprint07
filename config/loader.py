"""YAML + ENV configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

from stock_ai.config.settings import AppConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning new dict (immutable)."""
    result = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: str | Path = "config.yaml",
    env_path: str | Path = ".env",
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load AppConfig from YAML file + .env overrides.

    Args:
        config_path: Path to config.yaml (relative to CWD or absolute).
        env_path: Path to .env file.
        overrides: Additional dict overrides applied last.

    Returns:
        Fully validated AppConfig instance.
    """
    config_path = Path(config_path)
    env_path = Path(env_path)

    # Load .env if it exists (never fail if missing)
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(f"Loaded .env from {env_path}")
    else:
        logger.debug(f".env not found at {env_path}, skipping")

    # Load YAML config
    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        logger.debug(f"Loaded config from {config_path}")
    else:
        logger.warning(f"config.yaml not found at {config_path}, using defaults")

    # Apply overrides
    if overrides:
        raw = _deep_merge(raw, overrides)

    config = AppConfig(**raw)

    # Auto-disable Reddit if credentials missing
    reddit_id = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    if not (reddit_id and reddit_secret and
            reddit_id != "your_client_id_here" and
            reddit_secret != "your_client_secret_here"):
        # Return a new config with Reddit disabled
        crawlers_dict = config.crawlers.model_dump()
        crawlers_dict["reddit"]["enabled"] = False
        raw_updated = {**raw, "crawlers": crawlers_dict}
        config = AppConfig(**raw_updated)
        logger.info("Reddit credentials not found — Reddit crawler disabled")

    logger.info(
        f"Config loaded: ticker={config.ticker.default}, "
        f"market={config.ticker.market}, "
        f"start={config.dates.start}"
    )
    return config
