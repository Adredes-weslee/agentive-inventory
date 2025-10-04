"""Demand forecasting service based on the M5 dataset.

The implementation intentionally favours deterministic, lightweight
algorithms so that the API remains responsive even in restricted
execution environments.  A smoothed moving‑average baseline is used to
project demand for the requested horizon.  Prediction intervals are
derived from recent variability and aligned with the official M5
calendar to guarantee that forecast dates are valid.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Iterable, List

import numpy as np
import pandas as pd

from ..models.schemas import ForecastPoint, ForecastResponse

LOGGER = logging.getLogger(__name__)


class ForecastingService:
    """Generate per‑SKU forecasts from the M5 sales history."""

    def __init__(self, data_root: str = "data", moving_average_window: int = 28) -> None:
        self.data_root = data_root
        self.moving_average_window = moving_average_window
        self.sales_df: pd.DataFrame | None = None
        self.calendar_df: pd.DataFrame | None = None
        self._load_data()

    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        """Load the required CSV files into memory.

        The files are read once at service initialisation and cached for
        subsequent requests.  Missing files are tolerated here so that we
        can raise a user‑friendly error when a forecast is requested.
        """

        sales_path = os.path.join(self.data_root, "sales_train_validation.csv")
        calendar_path = os.path.join(self.data_root, "calendar.csv")

        if os.path.exists(sales_path):
            try:
                self.sales_df = pd.read_csv(sales_path)
            except Exception as exc:  # pragma: no cover - defensive branch
                LOGGER.exception("Failed to load sales dataset at %s", sales_path)
                raise FileNotFoundError(
                    "Unable to load sales_train_validation.csv; please verify the file integrity."
                ) from exc

        if os.path.exists(calendar_path):
            try:
                calendar_df = pd.read_csv(calendar_path)
                calendar_df["date"] = pd.to_datetime(calendar_df["date"], format="%Y-%m-%d")
                self.calendar_df = calendar_df
            except Exception as exc:  # pragma: no cover - defensive branch
                LOGGER.exception("Failed to load calendar dataset at %s", calendar_path)
                raise FileNotFoundError(
                    "Unable to load calendar.csv; please verify the file integrity."
                ) from exc

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self.sales_df is None or self.calendar_df is None:
            raise FileNotFoundError(
                "M5 dataset files were not found. Expected sales_train_validation.csv and calendar.csv under the data/ directory."
            )

    # ------------------------------------------------------------------
    def _get_sku_timeseries(self, sku_id: str) -> pd.Series:
        assert self.sales_df is not None
        sku_rows = self.sales_df[self.sales_df["item_id"] == sku_id]
        if sku_rows.empty:
            raise ValueError(f"SKU '{sku_id}' was not found in the sales dataset")

        demand_cols = [col for col in sku_rows.columns if col.startswith("d_")]
        if not demand_cols:
            raise ValueError(f"No demand history found for SKU '{sku_id}'")

        demand_series = sku_rows[demand_cols].sum(axis=0)
        demand_series.index = demand_cols
        return demand_series

    # ------------------------------------------------------------------
    def _map_days_to_dates(self, days: Iterable[str]) -> List[date]:
        assert self.calendar_df is not None
        calendar_map = dict(zip(self.calendar_df["d"], self.calendar_df["date"]))
        try:
            return [pd.to_datetime(calendar_map[d]).date() for d in days]
        except KeyError as exc:
            raise ValueError(f"Calendar is missing entries for day '{exc.args[0]}'") from exc

    # ------------------------------------------------------------------
    def _future_dates(self, last_day: str, horizon_days: int) -> List[date]:
        assert self.calendar_df is not None
        calendar_df = self.calendar_df
        try:
            last_idx = calendar_df.index[calendar_df["d"] == last_day][0]
        except IndexError as exc:
            raise ValueError(f"Calendar does not contain the last observed day '{last_day}'") from exc

        future_slice = calendar_df.iloc[last_idx + 1 : last_idx + 1 + horizon_days]
        if len(future_slice) < horizon_days:
            raise ValueError(
                "Requested horizon exceeds available calendar dates. Reduce horizon_days or provide extended calendar data."
            )
        return [d.date() for d in future_slice["date"]]

    # ------------------------------------------------------------------
    def forecast(self, sku_id: str, horizon_days: int = 28) -> ForecastResponse:
        """Return a per‑day forecast for the requested SKU.

        The routine relies solely on the historical sales for the SKU and
        generates a smoothed moving average baseline along with a simple
        prediction interval.  The interval width is determined from the
        recent coefficient of variation to provide a sensible guardrail
        for downstream services.
        """

        if horizon_days <= 0:
            raise ValueError("horizon_days must be a positive integer")

        self._ensure_loaded()
        assert self.sales_df is not None and self.calendar_df is not None

        demand_series = self._get_sku_timeseries(sku_id)
        dates = self._map_days_to_dates(demand_series.index)
        history = pd.Series(demand_series.values, index=pd.DatetimeIndex(dates, name="date"))
        history = history.sort_index()

        window = min(self.moving_average_window, len(history))
        if window == 0:
            raise ValueError(f"SKU '{sku_id}' does not have sufficient history to forecast")

        recent = history.iloc[-window:]
        mean_forecast = float(recent.mean())
        # Estimate variability: use rolling std, fall back to poisson like sqrt(mean)
        std_estimate = float(recent.std(ddof=1)) if len(recent) > 1 else float(np.sqrt(max(mean_forecast, 1.0)))
        if std_estimate == 0.0:
            std_estimate = float(np.sqrt(max(mean_forecast, 1.0)))

        LOGGER.info(
            "Forecasting SKU %s using SMA window=%s (mean=%.2f, std=%.2f, horizon=%s)",
            sku_id,
            window,
            mean_forecast,
            std_estimate,
            horizon_days,
        )

        future_dates = self._future_dates(demand_series.index[-1], horizon_days)

        # 80% interval (z ≈ 1.28155)
        z_score = 1.28155
        lo_val = max(mean_forecast - z_score * std_estimate, 0.0)
        hi_val = mean_forecast + z_score * std_estimate

        # Confidence scaled between 0.3 and 0.95 based on coefficient of variation
        coeff_var = std_estimate / mean_forecast if mean_forecast else float("inf")
        if np.isinf(coeff_var):
            confidence = 0.3
        else:
            confidence = float(max(0.3, min(0.95, 1.0 / (1.0 + coeff_var))))

        points: List[ForecastPoint] = [
            ForecastPoint(
                date=forecast_date,
                mean=round(mean_forecast, 4),
                lo=round(lo_val, 4),
                hi=round(hi_val, 4),
                model="sma",
                confidence=round(confidence, 4),
            )
            for forecast_date in future_dates
        ]

        return ForecastResponse(sku_id=sku_id, horizon_days=horizon_days, forecast=points)