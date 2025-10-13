"""API endpoints for reading and updating configuration YAML files."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, Optional

import yaml
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()

CONFIG_DIR = os.getenv("CONFIG_DIR", "configs")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.yaml")
THRESHOLDS_PATH = os.path.join(CONFIG_DIR, "thresholds.yaml")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _safe_write_yaml(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp-", suffix=".yaml", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        shutil.move(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


class SettingsUpdate(BaseModel):
    service_level_target: Optional[float] = Field(None, ge=0.5, le=0.999)
    carrying_cost_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    lead_time_days: Optional[int] = Field(None, ge=0, le=365)
    order_setup_cost: Optional[float] = Field(None, ge=0.0)
    default_unit_cost: Optional[float] = Field(None, ge=0.0)


class ThresholdsUpdate(BaseModel):
    auto_approval_limit: Optional[float] = Field(None, ge=0.0)
    min_service_level: Optional[float] = Field(None, ge=0.5, le=0.999)
    gmroi_min: Optional[float] = Field(None, ge=0.0, le=1.0)


def _merge_updates(original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = original.copy()
    result.update({k: v for k, v in updates.items() if v is not None})
    return result


@router.get("/configs/settings")
def get_settings() -> Dict[str, Any]:
    try:
        return _load_yaml(SETTINGS_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "not_found",
                "message": "settings.yaml not found",
            },
        ) from exc


@router.put("/configs/settings")
def put_settings(body: SettingsUpdate) -> Dict[str, Any]:
    try:
        current = _load_yaml(SETTINGS_PATH)
    except FileNotFoundError:
        current = {}

    updated = _merge_updates(current, body.model_dump(exclude_none=True))
    if updated == current:
        return current

    try:
        _safe_write_yaml(SETTINGS_PATH, updated)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "write_failed", "message": str(exc)},
        ) from exc
    return updated


@router.get("/configs/thresholds")
def get_thresholds() -> Dict[str, Any]:
    try:
        return _load_yaml(THRESHOLDS_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "not_found",
                "message": "thresholds.yaml not found",
            },
        ) from exc


@router.put("/configs/thresholds")
def put_thresholds(body: ThresholdsUpdate) -> Dict[str, Any]:
    try:
        current = _load_yaml(THRESHOLDS_PATH)
    except FileNotFoundError:
        current = {}

    updated = _merge_updates(current, body.model_dump(exclude_none=True))
    if updated == current:
        return current

    try:
        _safe_write_yaml(THRESHOLDS_PATH, updated)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "write_failed", "message": str(exc)},
        ) from exc
    return updated
