r"""backend\app\services\forecasting_service.py

Demand forecasting service built around the M5 competition dataset.

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
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, List, Sequence

import joblib
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

        self.model_default: str = "sma"
        self.model_portfolio: dict[str, str] = {"A": "xgb", "B": "prophet", "C": "sma"}
        self._abc_cache: dict[str, str] = {}
        self._abc_thresholds: tuple[float, float] | None = None

        data_root_path = Path(os.getenv("DATA_DIR", self.data_root))
        models_dir_env = os.getenv("MODELS_DIR")
        self.models_dir: Path = (
            Path(models_dir_env)
            if models_dir_env
            else data_root_path / "models"
        )
        self._model_cache: dict[str, object] = {}
        self._model_artifacts: dict[str, Path] = {}

        self._warm_model_cache()

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
            self._build_abc_cache()

    # ------------------------------------------------------------------
    def _build_abc_cache(self) -> None:
        assert self.sales_df is not None

        self._abc_cache.clear()
        self._abc_thresholds = None

        demand_cols = [c for c in self.sales_df.columns if c.startswith("d_")]
        if not demand_cols:
            return

        demand_frame = (
            self.sales_df[demand_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        averages = demand_frame.mean(axis=1)
        if averages.empty:
            return

        q80 = float(averages.quantile(0.8))
        q50 = float(averages.quantile(0.5))
        self._abc_thresholds = (q50, q80)

        id_column: str | None = None
        for candidate in ("id", "item_id"):
            if candidate in self.sales_df.columns:
                id_column = candidate
                break

        if id_column is not None:
            for sku, avg in zip(self.sales_df[id_column], averages):
                cls = self._class_from_average(float(avg))
                if cls:
                    self._abc_cache[str(sku)] = cls

        if "item_id" in self.sales_df.columns:
            grouped_totals = (
                demand_frame.assign(item_id=self.sales_df["item_id"])
                .groupby("item_id")
                .sum()
            )
            for item_id, avg in grouped_totals.mean(axis=1).items():
                cls = self._class_from_average(float(avg))
                if cls:
                    self._abc_cache[str(item_id)] = cls

    # ------------------------------------------------------------------
    def _class_from_average(self, average: float | None) -> str | None:
        thresholds = self._abc_thresholds
        if thresholds is None or average is None or np.isnan(average):
            return None

        q50, q80 = thresholds
        if average >= q80:
            return "A"
        if average >= q50:
            return "B"
        return "C"

    # ------------------------------------------------------------------
    def _average_demand_for_sku(self, sku_id: str) -> float | None:
        if self.sales_df is None:
            return None

        demand_cols = [c for c in self.sales_df.columns if c.startswith("d_")]
        if not demand_cols:
            return None

        sku_rows = pd.DataFrame()
        if "id" in self.sales_df.columns:
            sku_rows = self.sales_df[self.sales_df["id"] == sku_id]
            if sku_rows.empty and "item_id" in self.sales_df.columns:
                sku_rows = self.sales_df[self.sales_df["item_id"] == sku_id]
        elif "item_id" in self.sales_df.columns:
            sku_rows = self.sales_df[self.sales_df["item_id"] == sku_id]

        if sku_rows.empty:
            return None

        demand_history = sku_rows[demand_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if demand_history.empty:
            return None

        mean_per_row = demand_history.mean(axis=1)
        if mean_per_row.empty:
            return None

        return float(mean_per_row.mean())

    # ------------------------------------------------------------------
    def _abc_class_for_sku(self, sku_id: str, series: pd.Series | None = None) -> str | None:
        cached = self._abc_cache.get(sku_id)
        if cached:
            return cached

        average: float | None = None
        if series is not None and not series.empty:
            average = float(series.mean())
        else:
            average = self._average_demand_for_sku(sku_id)

        cls = self._class_from_average(average)
        if cls:
            self._abc_cache[sku_id] = cls
        return cls

    # ------------------------------------------------------------------
    def _preferred_model_for_sku(self, sku_id: str, series: pd.Series | None = None) -> str:
        abc_class = self._abc_class_for_sku(sku_id, series)
        preferred = self.model_portfolio.get(abc_class or "", self.model_default)
        if preferred not in {"sma", "prophet", "xgb"}:
            return self.model_default
        return preferred

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
        history.name = series.name

        model_class = self._preferred_model_for_sku(sku_id, series)

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
    def _warm_model_cache(self) -> None:
        """Ensure the model cache directory exists and record known artifacts."""

        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - filesystem failure
            LOGGER.exception("Unable to create model cache directory at %s", self.models_dir)
            return

        self._model_cache.clear()
        self._model_artifacts.clear()

        try:
            for artifact in self.models_dir.glob("*.joblib"):
                self._model_artifacts[artifact.stem] = artifact
        except OSError:  # pragma: no cover - filesystem failure
            LOGGER.exception("Unable to enumerate cached models in %s", self.models_dir)

    # ------------------------------------------------------------------
    def _model_cache_key(self, model_type: str, history: pd.Series) -> str:
        sku = (history.name or "series").replace(os.sep, "_")
        return f"{model_type}__{sku}"

    # ------------------------------------------------------------------
    def _model_artifact_path(self, cache_key: str) -> Path:
        path = self._model_artifacts.get(cache_key)
        if path is None:
            path = self.models_dir / f"{cache_key}.joblib"
            self._model_artifacts[cache_key] = path
        return path

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

        cache_key = self._model_cache_key("xgb", history)
        model_path = self._model_artifact_path(cache_key)
        cached = self._model_cache.get(cache_key)
        model: XGBRegressor | None = cached if isinstance(cached, XGBRegressor) else None
        trained = False

        if model is None and model_path.exists():
            try:
                loaded = joblib.load(model_path)
                model = loaded if isinstance(loaded, XGBRegressor) else None
            except Exception:  # pragma: no cover - defensive path
                LOGGER.exception("Failed to load cached XGBoost model at %s", model_path)
                model = None

        if model is None:
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
            trained = True

        self._model_cache[cache_key] = model
        if trained or not model_path.exists():
            try:
                joblib.dump(model, model_path)
            except Exception:  # pragma: no cover - defensive path
                LOGGER.exception("Failed to persist XGBoost model at %s", model_path)

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

        cache_key = self._model_cache_key("prophet", history)
        model_path = self._model_artifact_path(cache_key)
        cached = self._model_cache.get(cache_key)
        model: Prophet | None = cached if isinstance(cached, Prophet) else None
        trained = False

        if model is None and model_path.exists():
            try:
                loaded = joblib.load(model_path)
                model = loaded if isinstance(loaded, Prophet) else None
            except Exception:  # pragma: no cover - defensive path
                LOGGER.exception("Failed to load cached Prophet model at %s", model_path)
                model = None

        if model is None:
            df = pd.DataFrame({"ds": history.index, "y": history.values})
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
            )
            model.fit(df)
            trained = True

        self._model_cache[cache_key] = model
        if trained or not model_path.exists():
            try:
                joblib.dump(model, model_path)
            except Exception:  # pragma: no cover - defensive path
                LOGGER.exception("Failed to persist Prophet model at %s", model_path)

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

    def _candidate_models(self, history_length: int, preferred: str, strict: bool) -> List[str]:
        """Return the list of model candidates to evaluate."""

        preferred = preferred or "sma"
        candidates: List[str] = []

        def _available(model_name: str) -> bool:
            if model_name == "prophet" and not HAS_PROPHET:
                return False
            if model_name == "xgb" and not HAS_XGBOOST:
                return False
            return True

        if strict:
            return [preferred] if _available(preferred) else []

        if history_length < max(self.XGB_LAGS) * 2:
            priority = ["sma", preferred, "prophet"]
        else:
            priority = [preferred, "prophet", "xgb", "sma"]

        for candidate in priority:
            if candidate not in {"sma", "prophet", "xgb"}:
                continue
            if not _available(candidate):
                continue
            if candidate not in candidates:
                candidates.append(candidate)

        if "sma" not in candidates:
            candidates.append("sma")

        return candidates

    def _forecast_with_history(
        self,
        history: pd.Series,
        future_index: pd.DatetimeIndex,
        preferred_model: str,
        strict: bool,
    ) -> tuple[pd.Series, pd.Series, pd.Series, str]:
        """Return forecast components for an explicit history slice."""

        candidates = self._candidate_models(len(history), preferred_model, strict)
        if not candidates:
            raise ValueError(
                f"Requested model '{preferred_model}' is not available in the current environment."
            )

        last_error: ValueError | None = None

        for candidate in candidates:
            try:
                if candidate == "xgb":
                    mean_forecast, residuals, _ = self._xgb_forecast(history, future_index)
                    lower, upper = compute_pi(mean_forecast, residuals, self.z_value)
                elif candidate == "prophet":
                    (
                        mean_forecast,
                        lower,
                        upper,
                        _,
                        _,
                    ) = self._prophet_forecast(history, future_index)
                else:
                    mean_forecast, residuals, _ = self._sma_forecast(history, future_index)
                    lower, upper = compute_pi(mean_forecast, residuals, self.z_value)

                mean_forecast = mean_forecast.clip(lower=0.0)
                lower = lower.clip(lower=0.0)
                upper = upper.clip(lower=0.0)

                return mean_forecast, lower, upper, candidate
            except ValueError as exc:
                last_error = exc
                continue

        assert last_error is not None
        raise last_error

    def backtest(
        self,
        sku_id: str,
        window: int = 56,
        horizon: int = 28,
        step: int = 7,
        model_hint: str = "auto",
    ) -> dict[str, object]:
        """Run a rolling-origin backtest for the specified SKU."""

        if window <= 0 or horizon <= 0 or step <= 0:
            raise ValueError("window, horizon, and step must be positive integers")

        self._ensure_loaded()

        try:
            sku_history = self._sku_history(sku_id)
        except ValueError as exc:
            raise FileNotFoundError("M5 datasets appear to be missing required columns or entries.") from exc

        history = sku_history.series.sort_index()

        if len(history) < window + horizon:
            raise ValueError("Insufficient history for the requested window and horizon combination.")

        preferred_model = sku_history.model_class if model_hint == "auto" else model_hint
        strict = model_hint != "auto"

        dates: list[str] = []
        origin_dates: list[str] = []
        actual_values: list[float] = []
        predicted_values: list[float] = []
        coverage_hits = 0
        coverage_total = 0
        per_origin_coverage: list[float] = []
        model_used: str | None = None

        start_index = window
        last_start = len(history) - horizon

        while start_index <= last_start:
            train_slice = history.iloc[start_index - window : start_index]
            future_slice = history.iloc[start_index : start_index + horizon]
            future_index = future_slice.index

            if train_slice.empty or len(future_slice) < horizon:
                break

            mean_forecast, lower, upper, chosen_model = self._forecast_with_history(
                train_slice,
                pd.DatetimeIndex(future_index),
                preferred_model,
                strict,
            )

            if model_used is None:
                model_used = chosen_model

            origin_hits = 0
            origin_total = 0

            origin_dates.append(train_slice.index[-1].date().isoformat())

            for dt, actual in future_slice.items():
                prediction = float(mean_forecast.loc[dt])
                lo = float(lower.loc[dt]) if dt in lower else 0.0
                hi = float(upper.loc[dt]) if dt in upper else 0.0

                dates.append(dt.date().isoformat())
                actual_values.append(float(actual))
                predicted_values.append(prediction)

                coverage_total += 1
                if lo <= actual <= hi:
                    coverage_hits += 1
                    origin_hits += 1
                origin_total += 1

            if origin_total:
                per_origin_coverage.append(float(origin_hits / origin_total))

            start_index += step

        if not dates:
            raise ValueError("Unable to generate backtest windows with the provided parameters.")

        actual_arr = np.array(actual_values, dtype=float)
        pred_arr = np.array(predicted_values, dtype=float)
        mask = actual_arr != 0
        if mask.any():
            mape = float(np.mean(np.abs((actual_arr[mask] - pred_arr[mask]) / actual_arr[mask])))
        else:
            mape = 0.0

        coverage = float(coverage_hits / coverage_total) if coverage_total else 0.0

        history_tail = history.iloc[-window:]
        history_dates = [dt.date().isoformat() for dt in history_tail.index]
        history_values = [float(v) for v in history_tail.values]

        return {
            "dates": dates,
            "y": [float(v) for v in actual_values],
            "yhat": [float(v) for v in predicted_values],
            "mape": mape,
            "coverage": coverage,
            "per_origin_coverage": [float(v) for v in per_origin_coverage],
            "history_dates": history_dates,
            "history_values": history_values,
            "history": {
                "dates": history_dates,
                "y": history_values,
            },
            "origin_dates": origin_dates,
            "model_used": model_used or preferred_model,
        }

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

