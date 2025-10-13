from __future__ import annotations

import os
import logging
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status

LOGGER = logging.getLogger(__name__)
router = APIRouter()

DATA_DIR = os.getenv("DATA_DIR", "data")
SALES_PATH = os.path.join(DATA_DIR, "sales_train_validation.csv")

def _error(code: str, message: str) -> dict[str, str]:
    return {"error": code, "message": message}

@router.get("/catalog/ids")
def get_ids(limit: int = Query(20, ge=1, le=500)) -> dict[str, List[str]]:
    """Return up to `limit` M5 row ids for UI typeahead."""
    if not os.path.exists(SALES_PATH):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error("data_unavailable", f"{SALES_PATH} not found. Place M5 CSVs in ./data."),
        )
    try:
        df = pd.read_csv(SALES_PATH, usecols=["id"], nrows=limit)
        ids: List[str] = [str(x) for x in df["id"].dropna().tolist() if str(x).strip()]
        return {"ids": ids}
    except ValueError:
        LOGGER.warning("M5 sales file missing 'id' column")
        return {"ids": []}
    except Exception as exc:
        LOGGER.exception("Failed to read catalog ids: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error("catalog_error", "Unable to read catalog ids."),
        )
