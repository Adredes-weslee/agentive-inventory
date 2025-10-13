"""Routes for manual approvals and audit log retrieval."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

router = APIRouter()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
AUDIT_LOG_PATH = DATA_DIR / "audit_log.jsonl"
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000


class ApprovalRequest(BaseModel):
    """Validate incoming approval requests."""

    sku_id: str = Field(..., min_length=1, description="SKU identifier for the approval request")
    action: Literal["approve", "reject"] = Field(..., description="Approval decision for the SKU")
    qty: int | None = Field(None, ge=0, description="Quantity of units approved or rejected")
    reason: str | None = Field(None, description="Optional reason for the decision")


class ApprovalEvent(ApprovalRequest):
    """Persisted approval audit event."""

    ts: int = Field(..., ge=0, description="Unix timestamp when the event was recorded")


@router.post("/approvals", status_code=status.HTTP_201_CREATED)
async def create_approval(body: ApprovalRequest) -> dict[str, Any]:
    """Append an approval decision to the JSONL audit log."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event = ApprovalEvent(ts=int(time.time()), **body.model_dump())
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(event.model_dump_json(exclude_none=True))
        log_file.write("\n")

    LOGGER.info("Recorded approval event for sku_id=%s action=%s", event.sku_id, event.action)
    return {"ok": True, "event": event.model_dump()}


@router.get("/approvals/audit-log")
async def get_audit_log(
    limit: int = Query(
        DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description="Maximum number of most-recent audit events to return",
    )
) -> dict[str, list[dict[str, Any]]]:
    """Return the most recent approval events from the audit log."""

    if not AUDIT_LOG_PATH.exists():
        return {"events": []}

    recent_lines: deque[str] = deque(maxlen=limit)
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if line:
                recent_lines.append(line)

    events: list[dict[str, Any]] = []
    for line in reversed(recent_lines):  # newest first
        try:
            event = ApprovalEvent.model_validate_json(line)
        except ValueError:  # pragma: no cover - defensive: skip malformed entries
            LOGGER.warning("Skipping malformed audit log line: %s", line)
            continue
        events.append(event.model_dump())

    return {"events": events}
