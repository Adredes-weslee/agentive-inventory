"""Routes for demand forecasting."""

from fastapi import APIRouter, HTTPException, Query, status

from ..services.forecasting_service import ForecastingService, ForecastResponse

router = APIRouter()

_forecast_service = ForecastingService(data_root="data")


@router.get("/forecasts/{sku_id}", response_model=ForecastResponse)
async def get_forecast(
    sku_id: str,
    horizon_days: int = Query(28, ge=1, le=365, description="Forecast horizon in days"),
) -> ForecastResponse:
    """Return a forecast for the specified SKU."""

    try:
        return _forecast_service.forecast(sku_id=sku_id, horizon_days=int(horizon_days))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        status_code = status.HTTP_404_NOT_FOUND if "SKU" in message else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=message) from exc
