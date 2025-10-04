"""
Routes for demand forecasting.

Clients may request a forecast for a given SKU identifier.  The horizon
parameter determines how many days ahead to predict.  Forecasts are
returned as a list of data points including mean demand and prediction
intervals.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from ..services.forecasting_service import ForecastingService, ForecastPoint, ForecastResponse

router = APIRouter()

# Create a single forecasting service instance.  In a more complex application
# you might use dependency injection to provide this per-request.
_forecast_service = ForecastingService(data_root="data")


@router.get("/forecasts/{sku_id}", response_model=ForecastResponse)
async def get_forecast(sku_id: str, horizon_days: int = 28) -> ForecastResponse:
    """Return a forecast for the specified SKU.

    Args:
        sku_id: The item identifier from the M5 dataset (e.g. "HOBBIES_1_001").
        horizon_days: Number of future days to forecast (default: 28).

    Returns:
        A ``ForecastResponse`` containing the predicted demand and prediction
        intervals for each day in the horizon.
    """
    try:
        return _forecast_service.forecast(sku_id=sku_id, horizon_days=horizon_days)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))