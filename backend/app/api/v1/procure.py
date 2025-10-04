"""Routes for procurement recommendations."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..services.forecasting_service import ForecastingService, ForecastResponse
from ..services.procurement_service import ProcurementService, ReorderRec

router = APIRouter()

_forecast_service = ForecastingService(data_root="data")
_procurement_service = ProcurementService()


class ProcurementRequest(BaseModel):
    sku_id: str = Field(..., description="SKU identifier from the M5 dataset")
    horizon_days: Optional[int] = Field(28, ge=1, le=365)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


@router.post("/procure/recommendations", response_model=List[ReorderRec])
async def get_recommendations(body: ProcurementRequest) -> List[ReorderRec]:
    """Generate a purchase recommendation for a given SKU."""

    try:
        forecast: ForecastResponse = _forecast_service.forecast(
            sku_id=body.sku_id,
            horizon_days=int(body.horizon_days or 28),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        status_code = status.HTTP_404_NOT_FOUND if "SKU" in message else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=message) from exc

    recommendations = _procurement_service.recommend(forecast, body.context or {})
    return recommendations
