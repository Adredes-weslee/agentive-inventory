r"""backend\app\services\inventory_service.py

Inventory helper utilities built on top of the M5 dataset."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

import pandas as pd

try:  # pragma: no cover - optional dependency in minimal installs
    import pyarrow.dataset as ds
except Exception:  # pragma: no cover - gracefully degrade when pyarrow missing
    ds = None  # type: ignore[assignment]

from .io_utils import prefer_parquet

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SKUPrice:
    """Lightweight container for SKU price metadata."""

    item_id: str
    store_id: Optional[str]
    sell_price: float
    week: Optional[int]


class InventoryService:
    """Provide SKU metadata, unit cost estimates and inventory placeholders."""

    def __init__(self, data_root: str = "data", config_root: Optional[str] = None) -> None:
        self.data_root = Path(os.getenv("DATA_DIR", data_root))
        self.config_root = Path(config_root or "configs")
        self.sales_df: Optional[pd.DataFrame] = None
        self.prices_df: Optional[pd.DataFrame] = None
        self._prices_dataset: Optional["ds.Dataset"] = None
        self._settings_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def _sales_path(self) -> Path:
        return self.data_root / "sales_train_validation.csv"

    def _prices_path(self) -> Path:
        return self.data_root / "sell_prices.csv"

    def _prices_parquet_path(self) -> Path:
        return self._prices_path().with_suffix(".parquet")

    def _calendar_path(self) -> Path:
        return self.data_root / "calendar.csv"

    def data_files_present(self) -> bool:
        return self._sales_path().exists() and self._prices_path().exists() and self._calendar_path().exists()

    # ------------------------------------------------------------------
    def _load_settings(self) -> Dict[str, Any]:
        default_settings: Dict[str, Any] = {
            "lead_time_days": 7,
            "gross_margin_rate": 0.0,
            "lead_time_overrides": {},
        }

        settings_path = self.config_root / "settings.yaml"
        settings = default_settings.copy()

        if not settings_path.exists():
            LOGGER.debug("Settings file not found at %s; using defaults.", settings_path)
            return settings

        if yaml is None:
            LOGGER.warning("PyYAML is not installed; unable to read %s. Using defaults.", settings_path)
            return settings

        try:
            with open(settings_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
        except Exception as exc:  # pragma: no cover - defensive logging only
            LOGGER.warning("Unable to load settings from %s: %s", settings_path, exc)
            return settings

        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if value is not None:
                    settings[key] = value
        else:  # pragma: no cover - defensive logging only
            LOGGER.warning("Settings at %s are not a mapping; using defaults.", settings_path)

        return settings

    def _settings(self) -> Dict[str, Any]:
        if self._settings_cache is None:
            self._settings_cache = self._load_settings()
        return self._settings_cache

    # ------------------------------------------------------------------
    def load_sales(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        dtype: Optional[Dict[str, Any]] = None,
        **csv_kwargs: Any,
    ) -> pd.DataFrame:
        path = self._sales_path()
        if not path.exists():
            raise FileNotFoundError(f"Sales dataset not found at {path}")
        return prefer_parquet(path, columns=columns, dtype=dtype, **csv_kwargs)

    def load_prices(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        dtype: Optional[Dict[str, Any]] = None,
        **csv_kwargs: Any,
    ) -> pd.DataFrame:
        path = self._prices_path()
        if not path.exists():
            raise FileNotFoundError(f"Sell prices dataset not found at {path}")
        return prefer_parquet(path, columns=columns, dtype=dtype, **csv_kwargs)

    def _ensure_sales_df(self) -> Optional[pd.DataFrame]:
        if self.sales_df is not None:
            return self.sales_df

        try:
            self.sales_df = self.load_sales()
        except FileNotFoundError:
            self.sales_df = None
        except Exception as exc:  # pragma: no cover - defensive logging only
            LOGGER.warning("Unable to load sales dataset at %s: %s", self._sales_path(), exc)
            self.sales_df = None
        return self.sales_df

    def _load_prices_dataset(self) -> Optional["ds.Dataset"]:
        if ds is None:
            return None

        if self._prices_dataset is not None:
            return self._prices_dataset

        parquet_path = self._prices_parquet_path()
        csv_path = self._prices_path()

        try:
            if parquet_path.exists():
                self._prices_dataset = ds.dataset(parquet_path)
            elif csv_path.exists():
                self._prices_dataset = ds.dataset(str(csv_path), format="csv")
        except Exception:  # pragma: no cover - defensive logging only
            LOGGER.warning("Unable to initialise sell price dataset at %s", csv_path)
            self._prices_dataset = None

        return self._prices_dataset

    def _ensure_prices_df(self, *, allow_dataset: bool = True) -> Optional[pd.DataFrame]:
        if self.prices_df is not None:
            return self.prices_df

        # When pyarrow is available we prefer using dataset queries instead of
        # materialising the full table in memory. Only fall back to a pandas
        # frame when absolutely necessary.
        if allow_dataset and self._load_prices_dataset() is not None:
            return None

        try:
            prices = self.load_prices(
                columns=["item_id", "store_id", "wm_yr_wk", "sell_price"],
                dtype={
                    "item_id": "string",
                    "store_id": "string",
                    "wm_yr_wk": "int32",
                    "sell_price": "float32",
                },
            )
        except FileNotFoundError:
            prices = None
        except Exception as exc:  # pragma: no cover - defensive logging only
            LOGGER.warning("Unable to load sell prices dataset at %s: %s", self._prices_path(), exc)
            prices = None

        if prices is not None and not prices.empty and "wm_yr_wk" in prices.columns:
            with pd.option_context("mode.copy_on_write", True):
                prices = prices.copy()
                try:
                    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype(int)
                except Exception:  # pragma: no cover - dtype coercion guard
                    pass

        self.prices_df = prices
        return self.prices_df

    def _load_price_rows(
        self,
        item_id: str,
        store_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a filtered price frame for the requested identifiers."""

        dataset = self._load_prices_dataset()
        columns = ["item_id", "store_id", "wm_yr_wk", "sell_price"]

        if dataset is not None:
            expr: Optional["ds.Expression"] = None
            try:
                expr = ds.field("item_id") == item_id
                if store_id:
                    expr = expr & (ds.field("store_id") == store_id)
                available_columns = [col for col in columns if col in dataset.schema.names]
                table = dataset.to_table(columns=available_columns, filter=expr)
                if table is not None and table.num_rows:
                    frame = table.to_pandas()
                    return self._normalise_price_frame(frame)
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.warning(
                    "Falling back to pandas filtering for sell prices of %s/%s due to dataset error.",
                    item_id,
                    store_id or "*",
                )
                # Reset the dataset guard so the pandas path can materialise the
                # price frame instead of immediately returning ``None``.
                self._prices_dataset = None
                dataset = None

        prices_df = self._ensure_prices_df(allow_dataset=dataset is not None)
        if prices_df is None or prices_df.empty:
            return pd.DataFrame(columns=columns)

        mask = prices_df["item_id"] == item_id
        if store_id:
            mask &= prices_df["store_id"] == store_id

        frame = prices_df.loc[mask, columns]
        return self._normalise_price_frame(frame)

    def _normalise_price_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure price data uses numeric types for comparisons and sorting."""

        if frame.empty:
            return frame

        frame = frame.copy()
        expected = ["item_id", "store_id", "wm_yr_wk", "sell_price"]
        for column in expected:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame[expected]

        subset = [col for col in ("wm_yr_wk", "sell_price") if col in frame.columns]
        with pd.option_context("mode.copy_on_write", True):
            frame = frame.copy()
            for column in subset:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(subset=subset) if subset else frame

        dtype_map: dict[str, str | type] = {}
        if "wm_yr_wk" in frame.columns:
            dtype_map["wm_yr_wk"] = "int32"
        if "sell_price" in frame.columns:
            dtype_map["sell_price"] = "float32"

        return frame.astype(dtype_map) if dtype_map else frame

    # ------------------------------------------------------------------
    def has_sku(self, sku_id: str) -> bool:
        """Return ``True`` when the SKU exists in the sales dataset."""

        frame: Optional[pd.DataFrame] = self.sales_df

        if frame is None:
            try:
                frame = self.load_sales(columns=["id", "item_id"], dtype={"id": "string", "item_id": "string"})
            except FileNotFoundError:
                return False
            except Exception:  # pragma: no cover - defensive logging only
                LOGGER.warning("Unable to evaluate SKU existence for %s", sku_id)
                return False

        try:
            candidates = []
            if "id" in frame.columns:
                candidates.append(frame["id"].astype(str) == str(sku_id))
            if "item_id" in frame.columns:
                candidates.append(frame["item_id"].astype(str) == str(sku_id))
            return any(series.any() for series in candidates)
        except Exception:  # pragma: no cover - defensive logging only
            LOGGER.warning("Unable to evaluate SKU existence for %s", sku_id)
            return False

    # ------------------------------------------------------------------
    def sku_exists(self, sku_id: str) -> bool:
        return self.has_sku(sku_id)

    # ------------------------------------------------------------------
    def get_sku_info(self, sku_id: str) -> Dict[str, Any]:
        columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        df: Optional[pd.DataFrame] = self.sales_df
        if df is None:
            try:
                df = self.load_sales(columns=columns)
            except FileNotFoundError:
                return {}
            except Exception:  # pragma: no cover - defensive logging only
                LOGGER.warning("Unable to read SKU info for %s", sku_id)
                return {}

        row = pd.DataFrame()
        if "id" in df.columns:
            row = df[df["id"].astype(str) == str(sku_id)]
        if row.empty and "item_id" in df.columns:
            row = df[df["item_id"].astype(str) == str(sku_id)]
        if row.empty:
            return {}
        rec = row.iloc[0]
        return {
            "item_id": rec.get("item_id"),
            "dept_id": rec.get("dept_id"),
            "cat_id": rec.get("cat_id"),
            "store_id": rec.get("store_id"),
            "state_id": rec.get("state_id"),
        }

    # ------------------------------------------------------------------
    def get_current_inventory(self, sku_id: str) -> int:
        """Return stub on-hand inventory levels."""

        return 0

    # ------------------------------------------------------------------
    @lru_cache(maxsize=1024)
    def get_latest_price(self, sku_id: str) -> Optional[SKUPrice]:
        """Return the most recent sell price for the SKU."""

        frame = self._load_price_rows(sku_id)
        if frame.empty:
            return None

        latest = frame.sort_values("wm_yr_wk").iloc[-1]
        return SKUPrice(
            item_id=sku_id,
            store_id=latest.get("store_id"),
            sell_price=float(latest.get("sell_price", 0.0)),
            week=int(latest.get("wm_yr_wk")) if not pd.isna(latest.get("wm_yr_wk")) else None,
        )

    # ------------------------------------------------------------------
    def _get_price_series(self, sku_id: str, store_id: Optional[str] = None) -> Optional[pd.Series]:
        frame = self._load_price_rows(sku_id, store_id)
        if frame.empty and store_id:
            frame = self._load_price_rows(sku_id)

        if frame.empty:
            return pd.Series(dtype=float)

        series = frame.sort_values("wm_yr_wk")["sell_price"].astype(float)
        series.index = pd.RangeIndex(len(series))
        return series

    # ------------------------------------------------------------------
    def get_unit_cost(self, sku_id: str) -> float:
        """Return a proxy unit cost using median sell price data."""

        fallback_cost = 10.0
        store_id: Optional[str] = None
        lookup_item_id: Optional[str] = None

        # Prefer metadata from the sales table when available
        sku_info = self.get_sku_info(sku_id)
        if sku_info:
            lookup_item_id = sku_info.get("item_id") or None
            store_id = sku_info.get("store_id")

        # Fallback: parse composite identifiers from the SKU id itself
        if lookup_item_id is None:
            parsed_item, parsed_store = self._parse_ids(sku_id)
            lookup_item_id = parsed_item
            store_id = store_id or parsed_store

        # Price data is keyed by item identifier, not the composite row id
        price_series = self._get_price_series(lookup_item_id or sku_id, store_id)
        if price_series is None or price_series.empty:
            LOGGER.warning(
                "Falling back to default unit cost for SKU %s; median price unavailable.",
                sku_id,
            )
            return fallback_cost

        median_price = float(price_series.median())
        if median_price <= 0:
            LOGGER.warning(
                "Median sell price for SKU %s was non-positive; using fallback %.2f.",
                sku_id,
                fallback_cost,
            )
            return fallback_cost

        return median_price

    # ------------------------------------------------------------------
    def get_lead_time_days(self, sku_id: str) -> int:
        """Return supplier lead time with optional category overrides."""

        settings = self._settings()
        default_lead_time = int(settings.get("lead_time_days", 7))
        overrides = settings.get("lead_time_overrides", {})
        lead_time = default_lead_time

        if isinstance(overrides, dict) and overrides:
            sku_info = self.get_sku_info(sku_id)
            if sku_info:
                for key in ("item_id", "dept_id", "cat_id", "store_id"):
                    lookup_key = sku_id if key == "item_id" else sku_info.get(key)
                    if lookup_key is None:
                        continue
                    override = overrides.get(lookup_key)
                    if override is not None:
                        lead_time = int(override)
                        break

        if lead_time <= 0:
            LOGGER.warning(
                "Configured lead time for SKU %s was non-positive; using default %s",
                sku_id,
                default_lead_time,
            )
            lead_time = default_lead_time

        return lead_time

    # ------------------------------------------------------------------
    def get_price(self, sku_id: str) -> float:
        """Estimate a sell price using historical mean and configured margin uplift."""

        margin_rate = float(self._settings().get("gross_margin_rate", 0.0))
        store_id: Optional[str] = None
        lookup_item_id: Optional[str] = None

        sku_info = self.get_sku_info(sku_id)
        if sku_info:
            lookup_item_id = sku_info.get("item_id") or None
            store_id = sku_info.get("store_id")

        if lookup_item_id is None:
            parsed_item, parsed_store = self._parse_ids(sku_id)
            lookup_item_id = parsed_item
            store_id = store_id or parsed_store

        price_series = self._get_price_series(lookup_item_id or sku_id, store_id)
        if price_series is None or price_series.empty:
            LOGGER.warning(
                "Sell price data unavailable for SKU %s; deriving price from unit cost.",
                sku_id,
            )
            base_price = self.get_unit_cost(sku_id)
        else:
            base_price = float(price_series.mean())

        if base_price <= 0:
            base_price = 10.0
            LOGGER.warning(
                "Base price for SKU %s was non-positive; using fallback %.2f.",
                sku_id,
                base_price,
            )

        return base_price * (1 + margin_rate)

    # ------------------------------------------------------------------
    def estimate_unit_cost(self, sku_id: str) -> float:
        """Backward compatible wrapper for legacy callers."""

        return self.get_unit_cost(sku_id)

    # ------------------------------------------------------------------
    def _parse_ids(self, sku_id: str) -> tuple[Optional[str], Optional[str]]:
        """Parse item and store identifiers from composite SKU identifiers."""

        if not sku_id:
            return None, None
        parts = str(sku_id).split("_")
        item_id = "_".join(parts[:3]) if len(parts) >= 3 else None
        store_id = "_".join(parts[3:5]) if len(parts) >= 5 else None
        item_id = item_id or None
        store_id = store_id or None
        return item_id, store_id

    # ------------------------------------------------------------------
    def get_unit_price(self, sku_id: str) -> Optional[float]:
        """Return the median historical sell price for the SKU's item."""

        item_id, store_id = self._parse_ids(sku_id)

        if item_id is None:
            sku_info = self.get_sku_info(sku_id)
            item_id = sku_info.get("item_id") if sku_info else sku_id

        if item_id is None:
            return None

        frame = self._load_price_rows(item_id, store_id)
        if frame.empty and store_id:
            frame = self._load_price_rows(item_id)

        if frame.empty:
            return None

        median_price = float(frame["sell_price"].median())

        if not math.isfinite(median_price) or median_price <= 0:
            return None

        return median_price

    # ------------------------------------------------------------------
    def seasonality_multiplier(self, sku_id: str, horizon_days: int) -> float:
        """Estimate seasonal demand uplift/downlift for the forecast horizon."""

        if horizon_days <= 0:
            return 1.0

        df = self._ensure_sales_df()
        if df is None or df.empty:
            return 1.0
        row = pd.DataFrame()
        if "id" in df.columns:
            row = df[df["id"] == sku_id]

        if row.empty and "item_id" in df.columns:
            item_id, _ = self._parse_ids(sku_id)
            lookup_id = item_id or sku_id
            row = df[df["item_id"] == lookup_id]

        if row.empty:
            return 1.0

        value_cols = [col for col in row.columns if col.startswith("d_")]
        if not value_cols:
            return 1.0

        series = row[value_cols].T
        series.columns = ["y"]
        series["y"] = pd.to_numeric(series["y"], errors="coerce").fillna(0.0)

        start_date = pd.Timestamp("2011-01-29")
        dates = pd.date_range(start=start_date, periods=len(series), freq="D")
        series["month"] = dates.month

        overall_mean = float(series["y"].mean())
        if overall_mean <= 0:
            return 1.0

        month_index = (series.groupby("month")["y"].mean() / overall_mean).to_dict()
        if not month_index:
            return 1.0

        last_date = dates[-1]
        next_months = [
            (last_date + pd.Timedelta(days=offset)).month
            for offset in range(1, horizon_days + 1)
        ]
        multipliers = [month_index.get(month, 1.0) for month in next_months]
        multiplier = float(sum(multipliers) / len(multipliers))

        return max(0.5, min(1.5, multiplier))
