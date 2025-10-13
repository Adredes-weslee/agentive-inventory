"""Administrative endpoints for managing YAML configuration files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import APIRouter, HTTPException, status


router = APIRouter()

CONFIG_DIR = "configs"
_CONFIG_FILES = {"settings": "settings.yaml", "thresholds": "thresholds.yaml"}


def _config_path(name: str) -> Path:
    try:
        filename = _CONFIG_FILES[name]
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "config_not_found", "message": f"Unknown config '{name}'."},
        ) from exc

    base_dir = Path(CONFIG_DIR)
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)

    return base_dir / filename


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_config",
                "message": f"Configuration at '{os.fspath(path)}' must be a mapping.",
            },
        )
    return dict(data)


@router.get("/configs/{config_name}")
def get_config(config_name: str) -> Dict[str, Any]:
    """Return the contents of a configuration file as JSON."""

    path = _config_path(config_name)
    return _load_config(path)


@router.put("/configs/{config_name}")
def update_config(config_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``payload`` into the stored configuration and persist it."""

    path = _config_path(config_name)
    existing = _load_config(path)
    existing.update(payload)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(existing, handle, sort_keys=True)
    return existing
