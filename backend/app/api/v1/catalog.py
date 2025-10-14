r"""backend\app\api\v1\catalog.py"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query, status

from ...services.inventory_service import InventoryService

LOGGER = logging.getLogger(__name__)
router = APIRouter()

DATA_DIR = os.getenv("DATA_DIR", "data")
SALES_PATH = os.path.join(DATA_DIR, "sales_train_validation.csv")

_inventory_service = InventoryService()

DEFAULT_SAMPLE_IDS: List[str] = [
    "FOODS_1_001_CA_1_validation",
    "HOBBIES_1_002_CA_1_validation",
    "FOODS_3_090_CA_1_validation",
]

def _error(code: str, message: str) -> dict[str, str]:
    return {"error": code, "message": message}

@router.get("/catalog/ids")
def get_ids(limit: int = Query(20, ge=1, le=1000)) -> dict[str, List[str]]:
    """Return up to `limit` M5 row ids for UI typeahead."""
    sales_file = Path(SALES_PATH)
    if not sales_file.exists() and not sales_file.with_suffix(".parquet").exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error("data_unavailable", f"{SALES_PATH} not found. Place M5 CSVs in ./data."),
        )
    try:
        ids = _inventory_service.list_ids(limit)

        if ids:
            present = set(ids)
            for sample in DEFAULT_SAMPLE_IDS:
                if sample not in present:
                    ids.append(sample)
                    present.add(sample)

            priority = [sample for sample in DEFAULT_SAMPLE_IDS if sample in present]
            remainder = [item for item in ids if item not in priority]
            ordered = priority + remainder
        else:
            ordered = DEFAULT_SAMPLE_IDS.copy()

        return {"ids": ordered[:limit]}
    except ValueError:
        LOGGER.warning("M5 sales file missing 'id' column")
        return {"ids": []}
    except Exception as exc:
        LOGGER.exception("Failed to read catalog ids: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error("catalog_error", "Unable to read catalog ids."),
        )
