"""
Application configuration utilities.

This module defines the ``Settings`` class used throughout the service for
environment variables and provides helper functions to load YAML files
containing business rules and thresholds.
"""

from __future__ import annotations

import os
from functools import lru_cache

import yaml

try:  # pragma: no cover - import compatibility shim
    from pydantic_settings import BaseSettings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    try:
        from pydantic import BaseSettings  # type: ignore
    except Exception:  # pragma: no cover
        class BaseSettings:  # type: ignore[override]
            """Fallback BaseSettings implementation for tests.

            The minimal implementation stores provided keyword arguments as
            attributes and ignores environment variable loading.
            """

            def __init__(self, **data: object) -> None:
                for key, value in data.items():
                    setattr(self, key, value)


class Settings(BaseSettings):
    """Configuration loaded from environment variables or a .env file."""

    # API server configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # GEMINI API key (optional; required if using LLM features)
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")

    # Database connection URL (unused in this implementation but available)
    db_url: str = os.getenv("DB_URL", "sqlite:///./inventory.db")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=None)
def get_settings() -> Settings:
    """Return a cached instance of ``Settings``.

    Using a cache ensures that environment variables are only read once.
    """
    return Settings()


def load_yaml(file_path: str) -> dict:
    """Load a YAML file from the given path and return its contents.

    If the file does not exist, an empty dictionary is returned.
    """
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}