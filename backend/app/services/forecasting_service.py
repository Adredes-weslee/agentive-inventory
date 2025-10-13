"""Demand forecasting service built around the M5 competition dataset.

The implementation follows a tiered modelling strategy inspired by the
M5 forecasting competition.  Depending on the SKU importance and the
available third‑party libraries, the service can switch between a simple
statistical baseline, Facebook Prophet, or an XGBoost regressor with lag
features.  The module is intentionally self‑contained so that it can be
unit tested without loading the FastAPI app.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from math import sqrt
from statistics import NormalDist
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from prophet import Prophet

    HAS_PROPHET = True
except Exception:  # pragma: no cover - prophet not installed or unusable
    Prophet = None  # type: ignore
    HAS_PROPHET = False

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - xgboost not installed or unusable
    XGBRegressor = None  # type: ignore
    HAS_XGBOOST = False

from ..core.config import load_yaml
from ..models.schemas import ForecastPoint, ForecastResponse

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities (kept top-level for straightforward unit testing)


def build_lag_features(history: pd.Series, lags: Iterable[int]) -> pd.DataFrame:
    """Return a supervised learning frame with lagged demand features.

    Parameters
    ----------
    history:
        A ``pd.Series`` indexed by ``pd.DatetimeIndex`` containing the demand
        history of a single SKU.
    lags:
        Sequence of positive integers representing the lag offsets, in days.

    Returns
    -------
    ``pd.DataFrame`` with the target column ``y`` and one column per lag,
    e.g. ``lag_7``.  Rows that do not have all lag values are removed.
    """

    if history.empty:
        raise ValueError("history series must contain at least one observation")

    df = pd.DataFrame({"y": history.astype(float)})
    for lag in sorted(set(lags)):
        if lag <= 0:
            raise ValueError("lag offsets must be positive integers")
        df[f"lag_{lag}"] = df["y"].shift(lag)

    return df.dropna()


def choose_model(is_a_class: bool, has_prophet: bool, has_xgboost: bool) -> str:
    """Choose which forecasting model should be applied.

    The decision logic closely mirrors the project requirements:

    * Class-A SKUs should leverage the XGBoost model when available.
    * Otherwise Prophet is preferred for its richer seasonality support.
    * A simple moving average baseline is used as the safety fallback.
    """

    if is_a_class and has_xgboost:
        return "xgb"
    if has_prophet:
        return "prophet"
    return "sma"


def compute_pi(
    mean_forecast: pd.Series,
    residuals: pd.Series,
    z_value: float,
) -> tuple[pd.Series, pd.Series]:
    """Compute prediction interval bounds from residuals.

    Parameters
    ----------
    mean_forecast:
        Series containing the forecasted mean demand.
    residuals:
        Series of in-sample residuals (actual minus predicted).  The
        standard deviation of these residuals determines the interval width.
    z_value:
        Z-score derived from the desired service level.

    Returns
    -------
    Tuple of ``(lower, upper)`` bounds with non-negative values.
    """

    if residuals.empty:
        spread = sqrt(max(mean_forecast.mean(), 1.0))
    else:
        spread = float(residuals.std(ddof=1))
        if np.isnan(spread) or spread == 0.0:
            spread = sqrt(max(mean_forecast.mean(), 1.0))

    lower = (mean_forecast - z_value * spread).clip(lower=0.0)
    upper = (mean_forecast + z_value * spread).clip(lower=0.0)
    return lower, upper


# ---------------------------------------------------------------------------
# Result container


class ForecastResult(ForecastResponse):
    """Extended forecast response enriched with a pandas ``DataFrame`` view."""

    @property
    def yhat(self) -> pd.DataFrame:
        """Return the forecast as a ``pd.DataFrame`` with the canonical schema."""

        return pd.DataFrame(
            [
                {
                    "date": point.date,
                    "mean": float(point.mean),
                    "lo": float(point.lo),
                    "hi": float(point.hi),
                    "model": point.model,
                    "confidence": float(point.confidence),
                }
                for point in self.forecast
            ]
        )


# ---------------------------------------------------------------------------
# Core service implementation


@dataclass(slots=True)
class _SkuHistory:
    sku_id: str
    series: pd.Series
    model_class: str
    last_calendar_key: str


class ForecastingService:
    """Generate per-SKU forecasts leveraging the M5 historical data."""

    DEFAULT_SMA_WINDOW: int = 56
    XGB_LAGS: Sequence[int] = (7, 14, 28)

    def __init__(
        self,
        data_root: str = "data",
        config_root: str = "configs",
        moving_average_window: int | None = None,
    ) -> None:
        self.data_root = data_root
        self.config_root = config_root
        self.sma_window = int(moving_average_window or self.DEFAULT_SMA_WINDOW)
        if self.sma_window <= 0:
            raise ValueError("moving_average_window must be a positive integer")

        self.sales_df: pd.DataFrame | None = None
        self.calendar_df: pd.DataFrame | None = None
        self.sell_prices_df: pd.DataFrame | None = None
        self.calendar_map: dict[str, pd.Timestamp] | None = None

        self.service_level: float = 0.95
        self.z_value: float = NormalDist().inv_cdf(self.service_level)
        self._a_class_cutoff: float | None = None

        self._load_configuration()
        self._load_data()

    # ------------------------------------------------------------------
    def _load_configuration(self) -> None:
        settings_path = os.path.join(self.config_root, "settings.yaml")
        settings = load_yaml(settings_path)
        service_level = float(
            settings.get(
                "service_level_target",
                settings.get("default_service_level", self.service_level),
            )
        )
        service_level = min(max(service_level, 0.5), 0.995)
        self.service_level = service_level
        self.z_value = NormalDist().inv_cdf(service_level)

    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        sales_path = os.path.join(self.data_root, "sales_train_validation.csv")
        calendar_path = os.path.join(self.data_root, "calendar.csv")
        sell_price_path = os.path.join(self.data_root, "sell_prices.csv")

        if not os.path.exists(sales_path):
            LOGGER.warning("Sales dataset missing at %s", sales_path)
        else:
            self.sales_df = pd.read_csv(sales_path)

        if not os.path.exists(calendar_path):
            LOGGER.warning("Calendar dataset missing at %s", calendar_path)
        else:
            calendar_df = pd.read_csv(calendar_path)
            calendar_df["date"] = pd.to_datetime(calendar_df["date"], format="%Y-%m-%d")
            self.calendar_df = calendar_df
            self.calendar_map = dict(zip(calendar_df["d"], calendar_df["date"]))

        if os.path.exists(sell_price_path):
            try:
                sell_prices = pd.read_csv(sell_price_path)
                sell_prices["wm_yr_wk"] = sell_prices["wm_yr_wk"].astype(int)
                self.sell_prices_df = sell_prices
            except Exception:  # pragma: no cover - defensive path
                LOGGER.exception("Failed to load sell price dataset at %s", sell_price_path)

        if self.sales_df is not None:
            self._compute_a_class_cutoff()

    # ------------------------------------------------------------------
    def _compute_a_class_cutoff(self) -> None:
        assert self.sales_df is not None
        demand_cols = [c for c in self.sales_df.columns if c.startswith("d_")]
        if not demand_cols:
            return
        totals = self.sales_df[demand_cols].sum(axis=1)
        self._a_class_cutoff = float(np.quantile(totals, 0.8))

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self.sales_df is None or self.calendar_df is None or self.calendar_map is None:
            raise FileNotFoundError(
                "M5 dataset files were not found. Expected sales_train_validation.csv and calendar.csv under the data/ directory."
            )

    # ------------------------------------------------------------------
    def _resolve_sku(self, sku_id: str) -> pd.Series:
        assert self.sales_df is not None
        if "id" in self.sales_df.columns:
            sku_rows = self.sales_df[self.sales_df["id"] == sku_id]
            if sku_rows.empty and "item_id" in self.sales_df.columns:
                sku_rows = self.sales_df[self.sales_df["item_id"] == sku_id]
        else:
            sku_rows = self.sales_df[self.sales_df["item_id"] == sku_id]

        if sku_rows.empty:
            raise ValueError("SKU not found")

        demand_cols = [col for col in sku_rows.columns if col.startswith("d_")]
        if not demand_cols:
            raise ValueError(f"No demand history found for SKU '{sku_id}'")

        demand_history = sku_rows[demand_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if len(sku_rows) > 1:
            demand_series = demand_history.sum(axis=0)
        else:
            demand_series = demand_history.iloc[0]

        demand_series = demand_series.astype(float)
        sku_series = pd.Series(demand_series.values, index=demand_cols, name=sku_id)
        return sku_series.ffill().fillna(0.0)

    # ------------------------------------------------------------------
    def _sku_history(self, sku_id: str) -> _SkuHistory:
        series = self._resolve_sku(sku_id)
        dates = self._map_days_to_dates(series.index)
        history = pd.Series(series.values, index=pd.DatetimeIndex(dates, name="date"))

        is_a_class = False
        if self._a_class_cutoff is not None:
            total_demand = float(series.sum())
            is_a_class = total_demand >= self._a_class_cutoff

        model_class = choose_model(is_a_class, HAS_PROPHET, HAS_XGBOOST)
        return _SkuHistory(
            sku_id=sku_id,
            series=history.sort_index(),
            model_class=model_class,
            last_calendar_key=str(series.index[-1]),
        )

    # ------------------------------------------------------------------
    def _map_days_to_dates(self, days: Sequence[str]) -> List[pd.Timestamp]:
        assert self.calendar_map is not None
        try:
            return [self.calendar_map[d] for d in days]
        except KeyError as exc:  # pragma: no cover - defensive branch
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
    def _confidence_from_mape(self, actual: pd.Series, predicted: pd.Series) -> float:
        if actual.empty or predicted.empty:
            return 0.5

        aligned = pd.concat([actual.rename("actual"), predicted.rename("pred")], axis=1).dropna()
        if aligned.empty:
            return 0.5

        mask = aligned["actual"] != 0
        if mask.sum() == 0:
            return 0.5
        mape = float(np.mean(np.abs((aligned.loc[mask, "actual"] - aligned.loc[mask, "pred"]) / aligned.loc[mask, "actual"])))
        if np.isnan(mape):
            return 0.5
        return float(max(0.1, min(0.99, 1.0 - min(mape, 1.5))))

    # ------------------------------------------------------------------
    def _add_calendar_features(self, index: pd.DatetimeIndex, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()
        df["dow"] = index.dayofweek
        df["month"] = index.month
        df["weekofyear"] = index.isocalendar().week.astype(int)
        df["is_weekend"] = (index.dayofweek >= 5).astype(int)

        if self.calendar_df is not None:
            calendar_indexed = self.calendar_df.set_index("date")
            calendar_slice = calendar_indexed.reindex(index)
            if "event_name_1" in calendar_slice:
                df["has_event"] = (~calendar_slice["event_name_1"].isna()).astype(int)
            for snap_col in ["snap_CA", "snap_TX", "snap_WI"]:
                if snap_col in calendar_slice:
                    df[snap_col] = calendar_slice[snap_col].fillna(0).astype(int)
        return df.fillna(0)

    # ------------------------------------------------------------------
    def _xgb_forecast(
        self, history: pd.Series, future_index: pd.DatetimeIndex
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        assert HAS_XGBOOST and XGBRegressor is not None
        if len(history) < max(self.XGB_LAGS) + 1:
            raise ValueError("Insufficient history length for XGBoost model")

        supervised = build_lag_features(history, self.XGB_LAGS)
        if supervised.empty:
            raise ValueError("Unable to create lag features for XGBoost model")

        features = supervised.drop(columns=["y"])
        features = self._add_calendar_features(supervised.index, features)
        target = supervised["y"].astype(float)

        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbosity=0,
        )
        model.fit(features, target)

        in_sample_pred = pd.Series(model.predict(features), index=features.index)
        residuals = target - in_sample_pred

        # Iterative forecasting using lagged predictions
        forecasts = []
        history_extended = history.copy()
        for dt_index in future_index:
            lag_values = {f"lag_{lag}": history_extended.iloc[-lag] for lag in self.XGB_LAGS}
            feature_row = pd.DataFrame([lag_values], index=[dt_index])
            feature_row = self._add_calendar_features(pd.DatetimeIndex([dt_index]), feature_row)
            prediction = float(model.predict(feature_row)[0])
            prediction = max(prediction, 0.0)
            forecasts.append(prediction)
            history_extended = pd.concat([history_extended, pd.Series([prediction], index=[dt_index])])

        mean_forecast = pd.Series(forecasts, index=future_index)
        return mean_forecast, residuals, in_sample_pred

    # ------------------------------------------------------------------
    def _prophet_forecast(
        self, history: pd.Series, future_index: pd.DatetimeIndex
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        assert HAS_PROPHET and Prophet is not None
        df = pd.DataFrame({"ds": history.index, "y": history.values})
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df)
        horizon_days = len(future_index)
        future = model.make_future_dataframe(periods=horizon_days, freq="D")
        forecast_df = model.predict(future)
        forecast_df.set_index("ds", inplace=True)

        in_sample_pred = forecast_df.loc[history.index, "yhat"]
        residuals = history - in_sample_pred

        forecast_slice = forecast_df.reindex(future_index)
        mean_forecast = forecast_slice["yhat"].clip(lower=0.0)
        lower = forecast_slice["yhat_lower"].clip(lower=0.0)
        upper = forecast_slice["yhat_upper"].clip(lower=0.0)

        return mean_forecast, lower, upper, residuals, in_sample_pred

    # ------------------------------------------------------------------
    def _sma_forecast(
        self, history: pd.Series, future_index: pd.DatetimeIndex
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        window = min(self.sma_window, len(history))
        if window == 0:
            raise ValueError("Insufficient history to compute moving average forecast")

        recent = history.iloc[-window:]
        mean_value = float(recent.mean())
        mean_forecast = pd.Series([mean_value] * len(future_index), index=future_index)
        in_sample_pred = pd.Series([mean_value] * len(recent), index=recent.index)
        residuals = recent - in_sample_pred
        return mean_forecast, residuals, in_sample_pred

    def forecast(self, sku_id: str, horizon_days: int = 28) -> ForecastResult:
        """Return a per-day forecast for the requested ``sku_id``."""

        if horizon_days <= 0:
            raise ValueError("horizon_days must be a positive integer")

        self._ensure_loaded()
        sku_history = self._sku_history(sku_id)
        history = sku_history.series

        if history.isna().any():
            history = history.fillna(method="ffill").fillna(0.0)

        model_choice = sku_history.model_class
        LOGGER.info(
            "Forecasting SKU %s initial_model=%s horizon=%s", sku_id, model_choice, horizon_days
        )

        future_dates = self._future_dates(sku_history.last_calendar_key, horizon_days)
        future_index = pd.DatetimeIndex(future_dates)

        forecast_df: pd.DataFrame | None = None
        residuals = pd.Series(dtype=float)
        in_sample_pred = pd.Series(dtype=float)
        model_used = model_choice
        last_error: ValueError | None = None

        candidate_models: List[str] = []
        if len(history) < max(self.XGB_LAGS) * 2:
            priority = ["sma", model_choice, "prophet"]
        else:
            priority = [model_choice, "prophet", "sma"]
        for candidate in priority:
            if candidate == "prophet" and not HAS_PROPHET:
                continue
            if candidate == "xgb" and not HAS_XGBOOST:
                continue
            if candidate not in candidate_models:
                candidate_models.append(candidate)
        if "sma" not in candidate_models:
            candidate_models.append("sma")

        for candidate in candidate_models:
            try:
                if candidate == "xgb":
                    mean_forecast, residuals, in_sample_pred = self._xgb_forecast(history, future_index)
                    lower, upper = compute_pi(mean_forecast, residuals, self.z_value)
                elif candidate == "prophet":
                    (
                        mean_forecast,
                        lower,
                        upper,
                        residuals,
                        in_sample_pred,
                    ) = self._prophet_forecast(history, future_index)
                else:
                    mean_forecast, residuals, in_sample_pred = self._sma_forecast(history, future_index)
                    lower, upper = compute_pi(mean_forecast, residuals, self.z_value)

                forecast_df = pd.DataFrame({"mean": mean_forecast, "lo": lower, "hi": upper})
                model_used = candidate
                break
            except ValueError as exc:
                LOGGER.warning("Model %s failed for SKU %s: %s", candidate, sku_id, exc)
                last_error = exc

        if forecast_df is None:
            assert last_error is not None
            raise last_error

        forecast_df.index = forecast_df.index.normalize()
        forecast_df = forecast_df.clip(lower=0.0)

        if not in_sample_pred.empty:
            actual_for_conf = history.reindex(in_sample_pred.index).dropna()
            predicted_for_conf = in_sample_pred.reindex(actual_for_conf.index)
        else:
            actual_for_conf = history
            predicted_for_conf = pd.Series([history.mean()] * len(history), index=history.index)
        confidence = self._confidence_from_mape(actual_for_conf, predicted_for_conf)

        forecast_df["model"] = model_used
        forecast_df["confidence"] = confidence
        forecast_df.reset_index(inplace=True)
        forecast_df.rename(columns={"index": "date"}, inplace=True)
        forecast_df["date"] = forecast_df["date"].dt.date

        points: List[ForecastPoint] = [
            ForecastPoint(
                date=row["date"],
                mean=float(row["mean"]),
                lo=float(row["lo"]),
                hi=float(row["hi"]),
                model=str(row["model"]),
                confidence=float(row["confidence"]),
            )
            for _, row in forecast_df.iterrows()
        ]

        return ForecastResult(sku_id=sku_id, horizon_days=horizon_days, forecast=points)

