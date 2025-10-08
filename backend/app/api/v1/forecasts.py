"""Routes for demand forecasting."""

from __future__ import annotations

import logging
from typing import Callable, Iterable

from fastapi import APIRouter, HTTPException, Query, status

from ...models import schemas
from ...services.forecasting_service import ForecastingService
from ...services.inventory_service import InventoryService

LOGGER = logging.getLogger(__name__)

router = APIRouter()

MIN_FORECAST_HORIZON_DAYS = 7
MAX_FORECAST_HORIZON_DAYS = 90

_forecast_service = ForecastingService(data_root="data")
_inventory_service = InventoryService(data_root="data")


def _error_payload(code: str, message: str) -> dict[str, str]:
    """Return a standardised error payload."""

    return {"error": code, "message": message}


def _validate_sku(sku_id: str) -> None:
    """Ensure that the SKU exists before attempting a forecast."""

    has_sku: Callable[[str], bool]
    has_sku = getattr(_inventory_service, "has_sku", _inventory_service.sku_exists)
    if getattr(_inventory_service, "sales_df", None) is None:
        LOGGER.error("Inventory datasets missing while validating forecast request for sku_id=%s", sku_id)
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


def _parse_horizon(raw_horizon: int) -> int:
    """Validate and normalise the requested forecast horizon."""

    try:
        horizon = int(raw_horizon)
    except (TypeError, ValueError) as exc:  # pragma: no cover - FastAPI already coerces ints
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_payload("invalid_horizon", "horizon_days must be an integer."),
        ) from exc

    if horizon < MIN_FORECAST_HORIZON_DAYS or horizon > MAX_FORECAST_HORIZON_DAYS:
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
    return horizon


def _map_forecast_points(points: Iterable) -> list[schemas.ForecastPoint]:
    """Convert service forecast points to API schemas."""

    return [schemas.ForecastPoint.model_validate(point) for point in points]


@router.get("/forecasts/{sku_id}", response_model=schemas.ForecastResponse)
async def get_forecast(
    sku_id: str,
    horizon_days: int = Query(28, description="Forecast horizon in days"),
) -> schemas.ForecastResponse:
    """Return a demand forecast for the specified SKU."""

    LOGGER.info("Forecast request received for sku_id=%s horizon=%s", sku_id, horizon_days)
    _validate_sku(sku_id)
    horizon = _parse_horizon(horizon_days)

    try:
        forecast_result = _forecast_service.forecast(sku_id=sku_id, horizon_days=horizon)
    except FileNotFoundError as exc:
        LOGGER.exception("Forecasting failed due to missing dataset files for sku_id=%s", sku_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error_payload(
                "data_unavailable",
                "Required dataset files are missing. Please upload the M5 datasets and retry.",
            ),
        ) from exc
    except ValueError as exc:
        LOGGER.warning("Forecasting rejected for sku_id=%s: %s", sku_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_payload("invalid_request", str(exc)),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.exception("Unexpected error while forecasting sku_id=%s", sku_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_payload("forecast_failed", "An unexpected error occurred while forecasting."),
        ) from exc

    return schemas.ForecastResponse(
        sku_id=forecast_result.sku_id,
        horizon_days=forecast_result.horizon_days,
        forecast=_map_forecast_points(forecast_result.forecast),
    )
