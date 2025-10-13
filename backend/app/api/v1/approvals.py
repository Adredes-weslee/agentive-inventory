"""Endpoints for managing manual approval workflows."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator


router = APIRouter()

DATA_DIR = "data"
LOG_PATH = os.path.join(DATA_DIR, "approvals_audit_log.jsonl")


class ApprovalRequest(BaseModel):
    sku_id: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    qty: int = Field(..., ge=0)
    reason: str = Field(..., min_length=1)

    @field_validator("action")
    @classmethod
    def _normalise_action(cls, value: str) -> str:
        return value.lower()


def _ensure_storage(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_event(event: Dict[str, Any]) -> None:
    path = Path(LOG_PATH)
    _ensure_storage(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")


def _read_events(limit: int) -> List[Dict[str, Any]]:
    path = Path(LOG_PATH)
    if not path.exists():
        return []

    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit <= 0:
        return events
    return events[-limit:]


@router.post("/approvals")
def create_approval(request: ApprovalRequest) -> Dict[str, Any]:
    """Record an approval decision and append it to the audit log."""

    event = {
        "sku_id": request.sku_id,
        "action": request.action,
        "qty": int(request.qty),
        "reason": request.reason,
        "recorded_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_event(event)
    return {"status": "ok", "event": event}


@router.get("/approvals/audit-log")
def get_audit_log(limit: int = 50) -> Dict[str, List[Dict[str, Any]]]:
    """Return the most recent approval audit log entries."""

    if limit < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_limit", "message": "limit must be non-negative."},
        )
    events = _read_events(limit)
    return {"events": events}
