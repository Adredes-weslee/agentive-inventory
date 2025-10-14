r"""backend\app\models\schemas.py

Pydantic models used throughout the API.

These models serve as both request payload validators and response
serialisation schemas.  Using typed models ensures that clients and
servers agree on the structure of the data being exchanged.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    """A single point in a demand forecast."""

    date: date
    mean: float = Field(..., description="Predicted mean demand for the date")
    lo: float = Field(..., description="Lower bound of the prediction interval")
    hi: float = Field(..., description="Upper bound of the prediction interval")
    model: str = Field(..., description="The model used for this forecast point")
    confidence: float = Field(..., description="Confidence score between 0 and 1")


class ForecastResponse(BaseModel):
    """A forecast for a given SKU over a specified horizon."""

    sku_id: str
    horizon_days: int
    forecast: List[ForecastPoint]


class ReorderRec(BaseModel):
    """Recommendation for reordering a SKU."""

    sku_id: str
    reorder_point: int = Field(..., description="Inventory level at which to reorder")
    order_qty: int = Field(..., description="Suggested order quantity")
    gmroi_delta: float = Field(..., description="Expected change in GMROI from this order")
    confidence: float = Field(..., description="Confidence in the recommendation, 0–1")
    requires_approval: bool = Field(
        ..., description="Whether a human must approve this recommendation"
    )