r"""backend\app\api\v1\backtest.py

Backtesting routes for forecasting models."""

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
    detail: bool = Query(
        False,
        description="When true, include detailed history and origin metadata in the response.",
    ),
    cv: Literal["none", "store", "category"] = Query(
        "none",
        description=(
            "Optional peer cross-validation summary: 'store' samples SKUs in the same store; "
            "'category' samples SKUs in the same category."
        ),
    ),
) -> dict[str, object]:
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
    result["mape"] = _round_metric(mape)

    coverage = result.get("coverage")
    result["coverage"] = _round_metric(coverage)

    if cv != "none":
        sales_df = _inventory_service.sales_df
        info = _inventory_service.get_sku_info(sku_id)
        key_column = "store_id" if cv == "store" else "cat_id"
        key_value = info.get(key_column) if info else None
        if sales_df is not None and key_value is not None and key_column in sales_df.columns:
            peers_df = sales_df[sales_df[key_column] == key_value]
            id_column = "id" if "id" in peers_df.columns else "item_id" if "item_id" in peers_df.columns else None
            if id_column is not None and not peers_df.empty:
                skip_ids = {str(sku_id)}
                item_id = info.get("item_id")
                if item_id:
                    skip_ids.add(str(item_id))

                peer_ids = [
                    pid
                    for pid in peers_df[id_column].astype(str).unique().tolist()
                    if pid not in skip_ids
                ][:50]

                mapes: list[float] = []
                for peer_id in peer_ids:
                    try:
                        peer_result = _forecast_service.backtest(
                            sku_id=peer_id,
                            window=window,
                            horizon=horizon,
                            step=step,
                            model_hint=model,
                        )
                    except Exception:  # pragma: no cover - defensive sampling of peers
                        continue

                    peer_mape = peer_result.get("mape")
                    if isinstance(peer_mape, (int, float, np.floating)) and np.isfinite(
                        float(peer_mape)
                    ):
                        mapes.append(float(peer_mape))

                if mapes:
                    result["cv"] = {
                        "by": cv,
                        "n": len(mapes),
                        "mean_mape": float(np.round(float(np.mean(mapes)), 6)),
                    }

    if detail:
        history_payload = result.get("history")
        if not isinstance(history_payload, dict):
            history_dates = result.get("history_dates")
            history_values = result.get("history_values")
            if isinstance(history_dates, list) and isinstance(history_values, list):
                history_payload = {"dates": history_dates, "y": history_values}
            else:
                history_payload = None
        if history_payload is not None:
            result["history"] = history_payload

        origin_dates_value = result.get("origin_dates")
        if origin_dates_value is not None:
            result["origin_dates"] = origin_dates_value

    return result
def _round_metric(value: object) -> float | None:
    if isinstance(value, (int, float, np.floating)):
        return float(np.round(float(value), 6))
    return None
