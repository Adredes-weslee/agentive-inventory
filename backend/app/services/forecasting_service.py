"""
Forecasting service using the M5 dataset.

This service reads the M5 sales and calendar files from the ``data/`` directory
and provides a naive demand forecast for a given SKU.  The current
implementation uses a simple moving average of recent demand as a
placeholder; you can replace this with Prophet, XGBoost or any other
model as required.  The forecast is returned as a ``ForecastResponse``
containing a list of ``ForecastPoint`` instances.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import numpy as np

from ..models.schemas import ForecastPoint, ForecastResponse


class ForecastingService:
    def __init__(self, data_root: str = "data") -> None:
        self.data_root = data_root
        self.sales_df: pd.DataFrame | None = None
        self.calendar_df: pd.DataFrame | None = None
        self._load_data()

    def _load_data(self) -> None:
        """Load sales and calendar data from CSV files.

        If the files are missing, the corresponding data frames are left as
        ``None``.  An exception will be raised later when forecasting is
        requested without data.
        """
        sales_path = os.path.join(self.data_root, "sales_train_validation.csv")
        cal_path = os.path.join(self.data_root, "calendar.csv")
        if os.path.exists(sales_path):
            self.sales_df = pd.read_csv(sales_path)
        if os.path.exists(cal_path):
            self.calendar_df = pd.read_csv(cal_path)

    def _ensure_loaded(self) -> None:
        """Ensure that the required datasets are loaded."""
        if self.sales_df is None or self.calendar_df is None:
            raise FileNotFoundError(
                "Required dataset files (sales_train_validation.csv and calendar.csv) are missing from the data directory."
            )

    def forecast(self, sku_id: str, horizon_days: int = 28) -> ForecastResponse:
        """Compute a naive forecast for the given SKU.

        Args:
            sku_id: The item identifier from the M5 dataset.
            horizon_days: Number of days to forecast ahead.

        Returns:
            A ``ForecastResponse`` containing the forecast points.

        Raises:
            FileNotFoundError: If the dataset CSV files are not available.
            ValueError: If the SKU is not found in the sales dataset.
        """
        self._ensure_loaded()
        assert self.sales_df is not None and self.calendar_df is not None

        # Filter sales records for the requested SKU
        sku_rows = self.sales_df[self.sales_df["item_id"] == sku_id]
        if sku_rows.empty:
            raise ValueError(f"SKU '{sku_id}' not found in sales dataset")

        # Identify demand columns (d_1, d_2, …)
        demand_cols = [col for col in sku_rows.columns if col.startswith("d_")]
        # Sum demand across all stores for this SKU
        demand_series = sku_rows[demand_cols].sum(axis=0)
        # Map day indices to dates using the calendar file
        # The calendar file has columns: d, date, weekday, etc.
        day_to_date = dict(zip(self.calendar_df["d"], pd.to_datetime(self.calendar_df["date"])))
        # Build a time series DataFrame
        ts = pd.DataFrame({
            "date": [day_to_date[d] for d in demand_cols],
            "value": demand_series.values,
        }).sort_values("date")

        # Compute a simple moving average as the point forecast
        window = min(14, len(ts))  # window length: last 14 days or entire series
        last_mean = float(ts["value"].tail(window).mean()) if window > 0 else 0.0
        # Provide a constant forecast for the horizon
        start_date = ts["date"].iloc[-1] + timedelta(days=1)
        forecast_dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
        # Use ±10% interval and fixed confidence for demonstration
        points: List[ForecastPoint] = []
        for date_point in forecast_dates:
            mean = last_mean
            lo = mean * 0.9
            hi = mean * 1.1
            points.append(
                ForecastPoint(
                    date=date_point.date(),
                    mean=mean,
                    lo=lo,
                    hi=hi,
                    model="naive",
                    confidence=0.5,
                )
            )
        return ForecastResponse(sku_id=sku_id, horizon_days=horizon_days, forecast=points)