r"""backend/app/api/v1/procure.py

Routes for procurement recommendations."""

from __future__ import annotations

import logging
import math
import os
from numbers import Real
from typing import Dict, Iterable, List, Optional

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


def _generate_recommendations(sku_id: str, horizon: int) -> List[schemas.ReorderRec]:
    """Shared helper to compute procurement recommendations for a SKU."""

    _validate_sku(sku_id)

    try:
        forecast_result = _forecast_service.forecast(sku_id=sku_id, horizon_days=horizon)
    except FileNotFoundError as exc:
        LOGGER.exception("Procurement forecast failed due to missing dataset files for sku_id=%s", sku_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error_payload(
                "data_unavailable",
                "Required dataset files are missing. Please upload the M5 datasets and retry.",
            ),
        ) from exc
    except ValueError as exc:
        LOGGER.warning("Procurement request rejected for sku_id=%s: %s", sku_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_payload("invalid_request", str(exc)),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.exception("Unexpected error while generating forecast for sku_id=%s", sku_id)
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
        LOGGER.warning("Procurement recommendation rejected for sku_id=%s: %s", sku_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_payload("invalid_request", str(exc)),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.exception("Unexpected error while generating procurement recommendation for sku_id=%s", sku_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_error_payload(
                "procurement_failed",
                "An unexpected error occurred while creating procurement recommendations.",
            ),
        ) from exc

    return _map_recommendations(recommendations)


class ProcurementRequest(BaseModel):
    """Validated payload for procurement recommendation requests."""

    sku_id: str = Field(..., description="SKU identifier from the M5 dataset", min_length=1)
    horizon_days: int | None = Field(
        DEFAULT_HORIZON_DAYS,
        description="Forecast horizon in days used for procurement planning",
    )


class ExplanationRequest(BaseModel):
    """Validated payload for requesting recommendation explanations."""

    sku_id: str = Field(..., description="SKU identifier from the M5 dataset", min_length=1)
    horizon_days: int = Field(
        DEFAULT_HORIZON_DAYS,
        description="Forecast horizon in days used for procurement planning",
        ge=MIN_FORECAST_HORIZON_DAYS,
        le=MAX_FORECAST_HORIZON_DAYS,
    )
    recommendations: List[schemas.ReorderRec]


class BatchRequest(BaseModel):
    """Payload for requesting recommendations across multiple SKUs."""

    sku_ids: List[str] = Field(..., min_length=1, description="List of SKU identifiers")
    horizon_days: int | None = Field(
        DEFAULT_HORIZON_DAYS,
        description="Forecast horizon in days used for procurement planning",
    )
    cash_budget: Optional[float] = Field(
        None,
        ge=0,
        description="Optional cash budget used to select recommendations",
    )


class BatchRecommendation(schemas.ReorderRec):
    """Extended recommendation with selection metadata for batch responses."""

    selected: bool = Field(..., description="Whether the recommendation fits within the budget")
    total_spend: float = Field(..., description="Total spend implied by the recommendation")


class BatchRecommendationResponse(BaseModel):
    """Batch recommendation response payload."""

    recommendations: List[BatchRecommendation]
    total_spend_selected: float


@router.post("/procure/recommendations", response_model=List[schemas.ReorderRec])
async def get_recommendations(body: ProcurementRequest) -> List[schemas.ReorderRec]:
    """Generate purchase recommendations for a SKU using forecasted demand."""

    LOGGER.info(
        "Procurement recommendation request received for sku_id=%s horizon=%s",
        body.sku_id,
        body.horizon_days,
    )
    horizon = _parse_horizon(body.horizon_days)
    return _generate_recommendations(body.sku_id, horizon)


@router.post("/procure/recommendations/explain")
async def explain(body: ExplanationRequest) -> dict[str, str]:
    """Return an LLM-generated rationale for recommendations, if enabled."""

    LOGGER.info(
        "Procurement explanation request received for sku_id=%s horizon=%s", body.sku_id, body.horizon_days
    )

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_error_payload("feature_disabled", "Explanation service is disabled."),
        )

    config_dir = os.getenv("CONFIG_DIR", "configs")
    settings: dict[str, object]
    thresholds: dict[str, object]

    try:
        with open(os.path.join(config_dir, "settings.yaml"), "r", encoding="utf-8") as handle:
            settings = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        settings = {}
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.warning("Failed to load settings.yaml: %s", exc)
        settings = {}

    try:
        with open(os.path.join(config_dir, "thresholds.yaml"), "r", encoding="utf-8") as handle:
            thresholds = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        thresholds = {}
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.warning("Failed to load thresholds.yaml: %s", exc)
        thresholds = {}

    context = {
        "sku_id": body.sku_id,
        "horizon_days": body.horizon_days,
        "settings": settings,
        "thresholds": thresholds,
    }

    try:
        explanation = explain_recommendation(context, [rec.model_dump() for rec in body.recommendations])
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.exception("Explanation generation failed for sku_id=%s: %s", body.sku_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_error_payload("llm_error", "Explanation service failed."),
        ) from exc

    explanation_text = (explanation or "").strip() or (
        "Automated explanation was empty; please review the numeric rationale."
    )
    return {"explanation": explanation_text}


@router.post(
    "/procure/batch_recommendations",
    response_model=BatchRecommendationResponse,
)
async def batch_recommendations(body: BatchRequest) -> BatchRecommendationResponse:
    """Compute recommendations for multiple SKUs and apply optional budget selection."""

    LOGGER.info(
        "Batch procurement request received for %d SKUs horizon=%s budget=%s",
        len(body.sku_ids),
        body.horizon_days,
        body.cash_budget,
    )
    horizon = _parse_horizon(body.horizon_days)

    raw_recommendations: List[Dict[str, object]] = []
    for sku_id in body.sku_ids:
        for recommendation in _generate_recommendations(sku_id, horizon):
            unit_cost = float(_inventory_service.get_unit_cost(recommendation.sku_id))
            total_spend = float(recommendation.order_qty * unit_cost)
            payload = recommendation.model_dump()
            payload.update({
                "selected": False,
                "total_spend": total_spend,
            })
            raw_recommendations.append(payload)

    if not raw_recommendations:
        return BatchRecommendationResponse(recommendations=[], total_spend_selected=0.0)

    budget = float(body.cash_budget) if body.cash_budget is not None else None
    if budget is not None and not math.isfinite(budget):  # pragma: no cover - defensive branch
        budget = None

    if budget is None:
        for rec in raw_recommendations:
            rec["selected"] = True
        total_spend_selected = sum(
            _coerce_float(rec.get("total_spend", 0.0)) for rec in raw_recommendations
        )
    else:
        ranked = sorted(
            raw_recommendations,
            key=lambda rec: _coerce_float(rec.get("gmroi_delta", 0.0)),
            reverse=True,
        )
        running_spend = 0.0
        for rec in ranked:
            spend = max(_coerce_float(rec.get("total_spend", 0.0)), 0.0)
            if spend <= 0:
                rec["selected"] = True
                continue
            if running_spend + spend <= budget:
                rec["selected"] = True
                running_spend += spend
            else:
                rec["selected"] = False
                rec["requires_approval"] = True
        total_spend_selected = sum(
            _coerce_float(rec.get("total_spend", 0.0))
            for rec in raw_recommendations
            if bool(rec.get("selected"))
        )

    response_recommendations = [BatchRecommendation.model_validate(rec) for rec in raw_recommendations]
    return BatchRecommendationResponse(
        recommendations=response_recommendations,
        total_spend_selected=total_spend_selected,
    )
def _coerce_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
