"""
Routes for procurement recommendations.

This endpoint accepts a SKU identifier and optional context information
and returns recommended reorder quantities.  It uses the forecasting
service to obtain the demand forecast and then applies business guardrails
to compute reorder points and quantities.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from ..services.forecasting_service import ForecastingService, ForecastResponse
from ..services.procurement_service import ProcurementService, ReorderRec

router = APIRouter()

_forecast_service = ForecastingService(data_root="data")
_procurement_service = ProcurementService()


class ProcurementRequest(BaseModel):
    sku_id: str
    horizon_days: Optional[int] = 28
    context: Optional[Dict[str, Any]] = None


@router.post("/procure/recommendations", response_model=List[ReorderRec])
async def get_recommendations(body: ProcurementRequest) -> List[ReorderRec]:
    """Generate a purchase recommendation for a given SKU.

    A forecast will be computed using the forecasting service; then
    the procurement service will decide how much to reorder based
    on configured guardrails.
    """
    try:
        forecast: ForecastResponse = _forecast_service.forecast(
            sku_id=body.sku_id, horizon_days=body.horizon_days or 28
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _procurement_service.recommend(forecast, body.context or {})