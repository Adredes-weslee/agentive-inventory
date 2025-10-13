"""Routes for procurement recommendations."""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List

import yaml
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ...models import schemas
from ...services.forecasting_service import ForecastingService
from ...services.inventory_service import InventoryService
from ...services.llm_service import explain_recommendation
from ...services.procurement_service import ProcurementService

LOGGER = logging.getLogger(__name__)

router = APIRouter()

MIN_FORECAST_HORIZON_DAYS = 7
MAX_FORECAST_HORIZON_DAYS = 90
DEFAULT_HORIZON_DAYS = 28

_forecast_service = ForecastingService(data_root="data")
_procurement_service = ProcurementService()
_inventory_service = InventoryService(data_root="data")


def _load_yaml_config(filename: str) -> Dict[str, object]:
    path = os.path.join("configs", filename)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:  # pragma: no cover - defensive path
        LOGGER.exception("Failed to read configuration file at %s", path)
        return {}
    if not isinstance(data, dict):
        return {}
    return dict(data)


def _error_payload(code: str, message: str) -> dict[str, str]:
    """Return a standardised error payload."""

    return {"error": code, "message": message}


def _validate_sku(sku_id: str) -> None:
    """Ensure the SKU exists before processing recommendations."""

    has_sku = getattr(_inventory_service, "has_sku", _inventory_service.sku_exists)
    if getattr(_inventory_service, "sales_df", None) is None:
        LOGGER.error(
            "Inventory datasets missing while validating procurement request for sku_id=%s",
            sku_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=
            _error_payload(
                "data_unavailable",
                "Required dataset files are missing. Please upload the M5 datasets and retry.",
            ),
        )
    if not has_sku(sku_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_error_payload("sku_not_found", f"SKU '{sku_id}' was not found in inventory."),
        )


def _parse_horizon(raw_horizon: int | None) -> int:
    """Validate the requested horizon for procurement calculations."""

    horizon = raw_horizon if raw_horizon is not None else DEFAULT_HORIZON_DAYS
    try:
        horizon_int = int(horizon)
    except (TypeError, ValueError) as exc:  # pragma: no cover - FastAPI coerces ints
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_payload("invalid_horizon", "horizon_days must be an integer."),
        ) from exc

    if horizon_int < MIN_FORECAST_HORIZON_DAYS or horizon_int > MAX_FORECAST_HORIZON_DAYS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=
            _error_payload(
                "invalid_horizon",
                (
                    "horizon_days must be between "
                    f"{MIN_FORECAST_HORIZON_DAYS} and {MAX_FORECAST_HORIZON_DAYS} days."
                ),
            ),
        )
    return horizon_int


def _map_recommendations(recommendations: Iterable) -> List[schemas.ReorderRec]:
    """Convert service recommendations into API schemas."""

    return [schemas.ReorderRec.model_validate(rec) for rec in recommendations]


class ProcurementRequest(BaseModel):
    """Validated payload for procurement recommendation requests."""

    sku_id: str = Field(..., description="SKU identifier from the M5 dataset", min_length=1)
    horizon_days: int | None = Field(
        DEFAULT_HORIZON_DAYS,
        description="Forecast horizon in days used for procurement planning",
    )


class ExplainRequest(BaseModel):
    """Payload for LLM-backed explanation requests."""

    sku_id: str = Field(..., min_length=1)
    horizon_days: int = Field(..., ge=1)
    recommendations: List[Dict[str, object]] = Field(default_factory=list)


@router.post("/procure/recommendations", response_model=List[schemas.ReorderRec])
async def get_recommendations(body: ProcurementRequest) -> List[schemas.ReorderRec]:
    """Generate purchase recommendations for a SKU using forecasted demand."""

    LOGGER.info(
        "Procurement recommendation request received for sku_id=%s horizon=%s",
        body.sku_id,
        body.horizon_days,
    )
    _validate_sku(body.sku_id)
    horizon = _parse_horizon(body.horizon_days)

    try:
        forecast_result = _forecast_service.forecast(sku_id=body.sku_id, horizon_days=horizon)
    except FileNotFoundError as exc:
        LOGGER.exception("Procurement forecast failed due to missing dataset files for sku_id=%s", body.sku_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error_payload(
                "data_unavailable",
                "Required dataset files are missing. Please upload the M5 datasets and retry.",
            ),
        ) from exc
    except ValueError as exc:
        LOGGER.warning("Procurement request rejected for sku_id=%s: %s", body.sku_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_payload("invalid_request", str(exc)),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.exception("Unexpected error while generating forecast for sku_id=%s", body.sku_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_payload("forecast_failed", "An unexpected error occurred while forecasting."),
        ) from exc

    forecast_response = schemas.ForecastResponse(
        sku_id=forecast_result.sku_id,
        horizon_days=forecast_result.horizon_days,
        forecast=[schemas.ForecastPoint.model_validate(point) for point in forecast_result.forecast],
    )

    try:
        recommendations = _procurement_service.recommend(forecast_response, context={})
    except ValueError as exc:
        LOGGER.warning("Procurement recommendation rejected for sku_id=%s: %s", body.sku_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_payload("invalid_request", str(exc)),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.exception("Unexpected error while generating procurement recommendation for sku_id=%s", body.sku_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_payload(
                "procurement_failed",
                "An unexpected error occurred while creating procurement recommendations.",
            ),
        ) from exc

    return _map_recommendations(recommendations)


@router.post("/procure/recommendations/explain")
async def explain_recommendations(body: ExplainRequest) -> Dict[str, str]:
    """Return a natural language explanation for the provided recommendations."""

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM integration disabled")

    context = {
        "sku_id": body.sku_id,
        "horizon_days": body.horizon_days,
        "settings": _load_yaml_config("settings.yaml"),
        "thresholds": _load_yaml_config("thresholds.yaml"),
    }
    explanation = explain_recommendation(context, body.recommendations)
    return {"explanation": explanation}
