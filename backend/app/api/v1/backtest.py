"""Backtesting routes for forecasting models."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status

from .forecasts import _error_payload
from ...services.forecasting_service import ForecastingService
from ...services.inventory_service import InventoryService

LOGGER = logging.getLogger(__name__)

router = APIRouter()

_forecast_service = ForecastingService(data_root="data")
_inventory_service = InventoryService(data_root="data")


@router.get("/backtest/{sku_id}")
def backtest(
    sku_id: str,
    window: int = Query(56, ge=28, le=365, description="Length of the rolling training window."),
    horizon: int = Query(28, ge=7, le=60, description="Forecast horizon used for evaluation."),
    step: int = Query(7, ge=1, le=28, description="Step size (in days) between backtest windows."),
    model: Literal["auto", "sma", "prophet", "xgb"] = Query(
        "auto", description="Model to evaluate. 'auto' selects the recommended model."
    ),
):
    """Return rolling forecast accuracy metrics for the requested SKU."""

    if not _inventory_service.has_sku(sku_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_error_payload("sku_not_found", f"SKU '{sku_id}' was not found in inventory."),
        )

    try:
        result = _forecast_service.backtest(
            sku_id=sku_id,
            window=window,
            horizon=horizon,
            step=step,
            model_hint=model,
        )
    except FileNotFoundError as exc:
        LOGGER.exception("Backtest failed due to missing dataset files for sku_id=%s", sku_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_error_payload("data_missing", str(exc)),
        ) from exc
    except ValueError as exc:
        LOGGER.warning("Backtest rejected for sku_id=%s: %s", sku_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_payload("invalid_request", str(exc)),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.exception("Unexpected error during backtest for sku_id=%s", sku_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_payload("backtest_failed", "An unexpected error occurred while running the backtest."),
        ) from exc

    mape = result.get("mape")
    result["mape"] = float(np.round(mape, 6)) if mape is not None and np.isfinite(mape) else None

    coverage = result.get("coverage")
    result["coverage"] = (
        float(np.round(coverage, 6)) if coverage is not None and np.isfinite(coverage) else None
    )
    return result
